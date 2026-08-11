import asyncio

import pytest
from pydantic import BaseModel, ConfigDict

from modelcore.application.tools import ToolExecutor, ToolGeneration, ToolRegistry
from modelcore.exceptions import ToolExecutionError, ToolNotFoundError, ToolRoundLimitError, ToolValidationError
from modelcore.models import ChatRequest, Message
from modelcore.models.tools import ToolCall, ToolCallingResponse, ToolDefinition


class WeatherArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    city: str


async def weather(city: str) -> dict[str, str]:
    return {"city": city, "weather": "sunny"}


def definition() -> ToolDefinition:
    return ToolDefinition("weather", "Get weather", WeatherArgs, weather)


def request() -> ChatRequest:
    return ChatRequest([Message.user("weather in Curitiba")], model="test")


def response(*calls: ToolCall) -> ToolCallingResponse:
    return ToolCallingResponse(None, calls, "test", "fake", None, "tool_calls")


def test_registry_is_explicit_and_rejects_duplicates_or_unknown_tools() -> None:
    registry = ToolRegistry([definition()])
    assert registry.get("weather").json_schema == WeatherArgs.model_json_schema()
    with pytest.raises(ValueError, match="already registered"):
        registry.register(definition())
    with pytest.raises(ToolNotFoundError):
        registry.get("missing")


@pytest.mark.asyncio
async def test_executor_validates_arguments_and_normalizes_async_result() -> None:
    executor = ToolExecutor(ToolRegistry([definition()]))
    result = await executor.execute(ToolCall("call-1", "weather", {"city": "Curitiba"}))
    assert result.name == "weather"
    assert result.tool_call_id == "call-1"
    assert result.content == '{"city":"Curitiba","weather":"sunny"}'


@pytest.mark.asyncio
async def test_executor_rejects_invalid_arguments_and_sanitizes_execution_errors() -> None:
    executor = ToolExecutor(ToolRegistry([definition()]))
    with pytest.raises(ToolValidationError):
        await executor.execute(ToolCall(None, "weather", {"unexpected": "x"}))

    def broken(city: str) -> str:
        raise RuntimeError("secret detail")

    broken_definition = ToolDefinition("broken", "Broken", WeatherArgs, broken)
    with pytest.raises(ToolExecutionError, match="failed"):
        await ToolExecutor(ToolRegistry([broken_definition])).execute(ToolCall(None, "broken", {"city": "x"}))


class FakeToolProvider:
    def __init__(self, first: ToolCallingResponse, second: ToolCallingResponse) -> None:
        self.first, self.second = first, second
        self.calls: list[tuple[str, object]] = []

    async def generate_with_tools(self, request: ChatRequest, tools: tuple[ToolDefinition, ...]) -> ToolCallingResponse:
        self.calls.append(("first", tools))
        return self.first

    async def continue_with_tool_results(
        self,
        request: ChatRequest,
        initial: ToolCallingResponse,
        results: tuple[object, ...],
        tools: tuple[ToolDefinition, ...],
    ) -> ToolCallingResponse:
        self.calls.append(("second", results))
        return self.second


@pytest.mark.asyncio
async def test_tool_generation_executes_exactly_one_round_then_returns_final_response() -> None:
    call = ToolCall("call-1", "weather", {"city": "Curitiba"})
    final = ToolCallingResponse("It is sunny", (), "test", "fake", None, "stop")
    provider = FakeToolProvider(response(call), final)

    result = await ToolGeneration(provider, ToolExecutor(ToolRegistry([definition()]))).generate(
        request(), [definition()]
    )

    assert result is final
    assert [name for name, _ in provider.calls] == ["first", "second"]


@pytest.mark.asyncio
async def test_tool_generation_returns_normal_response_without_executing_tools() -> None:
    final = ToolCallingResponse("No tool needed", (), "test", "fake", None, "stop")
    provider = FakeToolProvider(final, final)

    result = await ToolGeneration(provider, ToolExecutor(ToolRegistry([definition()]))).generate(
        request(), [definition()]
    )

    assert result is final
    assert [name for name, _ in provider.calls] == ["first"]


@pytest.mark.asyncio
async def test_tool_generation_rejects_second_round_and_propagates_cancellation() -> None:
    call = ToolCall("call-1", "weather", {"city": "Curitiba"})
    provider = FakeToolProvider(response(call), response(call))
    generation = ToolGeneration(provider, ToolExecutor(ToolRegistry([definition()])))
    with pytest.raises(ToolRoundLimitError):
        await generation.generate(request(), [definition()])

    async def cancelled(city: str) -> str:
        raise asyncio.CancelledError()

    cancelled_def = ToolDefinition("cancelled", "cancelled", WeatherArgs, cancelled)
    cancelled_provider = FakeToolProvider(response(ToolCall(None, "cancelled", {"city": "x"})), response())
    with pytest.raises(asyncio.CancelledError):
        await ToolGeneration(cancelled_provider, ToolExecutor(ToolRegistry([cancelled_def]))).generate(
            request(), [cancelled_def]
        )
