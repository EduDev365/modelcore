import asyncio
from dataclasses import fields

import pytest

from modelcore.application.fallback import FallbackProvider
from modelcore.application.telemetry import (
    LoggingTelemetrySink,
    TelemetryProvider,
)
from modelcore.exceptions.provider import ProviderUnavailableError
from modelcore.models.chat_request import ChatRequest
from modelcore.models.chat_response import ChatResponse
from modelcore.models.chat_stream_chunk import ChatStreamChunk
from modelcore.models.message import Message
from modelcore.models.telemetry import GenerationTelemetryEvent
from modelcore.models.usage import Usage


class SequenceClock:
    def __init__(self, *values: float) -> None:
        self._values = iter(values)

    def __call__(self) -> float:
        return next(self._values)


class CollectingSink:
    def __init__(self) -> None:
        self.events: list[GenerationTelemetryEvent] = []

    async def emit(self, event: GenerationTelemetryEvent) -> None:
        self.events.append(event)


class FailingSink:
    async def emit(self, event: GenerationTelemetryEvent) -> None:
        raise RuntimeError("sink unavailable")


class FakeLogger:
    def __init__(self) -> None:
        self.records: list[tuple[str, str, dict[str, object]]] = []

    def info(self, message: str, **kwargs: object) -> None:
        self.records.append(("info", message, kwargs))

    def warning(self, message: str, **kwargs: object) -> None:
        self.records.append(("warning", message, kwargs))


class FakeProvider:
    def __init__(self, outcomes: list[ChatResponse | BaseException]) -> None:
        self._outcomes = outcomes
        self.generate_calls = 0
        self.stream_calls = 0

    async def generate(self, request: ChatRequest) -> ChatResponse:
        outcome = self._outcomes[self.generate_calls]
        self.generate_calls += 1
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    async def _stream(self):
        yield ChatStreamChunk(content_delta="chunk", model="test-model", provider="fake")

    def stream(self, request: ChatRequest):
        self.stream_calls += 1
        return self._stream()


def make_request() -> ChatRequest:
    return ChatRequest(messages=[Message.user("secret prompt")], model="test-model", temperature=0.2)


def make_response(provider: str = "openai", usage: Usage | None = Usage(3, 2)) -> ChatResponse:
    return ChatResponse(
        content="secret generated content",
        model="test-model",
        provider=provider,
        usage=usage,
        finish_reason="stop",
    )


@pytest.mark.asyncio
async def test_success_emits_exact_safe_event_and_preserves_response() -> None:
    response = make_response()
    provider = FakeProvider([response])
    sink = CollectingSink()
    telemetry = TelemetryProvider(provider, sink, clock=SequenceClock(10.0, 10.25))

    result = await telemetry.generate(make_request())

    assert result is response
    assert sink.events == [
        GenerationTelemetryEvent(
            operation="generate",
            provider="openai",
            model="test-model",
            duration_ms=250.0,
            success=True,
            usage=Usage(3, 2),
            error_type=None,
        )
    ]


@pytest.mark.asyncio
async def test_success_without_usage_keeps_usage_absent() -> None:
    sink = CollectingSink()
    telemetry = TelemetryProvider(FakeProvider([make_response(usage=None)]), sink, clock=SequenceClock(1, 2))

    await telemetry.generate(make_request())

    assert sink.events[0].usage is None


@pytest.mark.asyncio
async def test_failure_emits_safe_event_and_propagates_original_exception() -> None:
    error = ProviderUnavailableError("connection failed")
    sink = CollectingSink()
    telemetry = TelemetryProvider(FakeProvider([error]), sink, clock=SequenceClock(1, 1.5))

    with pytest.raises(ProviderUnavailableError) as raised:
        await telemetry.generate(make_request())

    assert raised.value is error
    assert sink.events == [
        GenerationTelemetryEvent(
            operation="generate",
            provider=None,
            model="test-model",
            duration_ms=500.0,
            success=False,
            usage=None,
            error_type="ProviderUnavailableError",
        )
    ]


@pytest.mark.asyncio
async def test_cancelled_error_propagates_without_emitting_telemetry() -> None:
    sink = CollectingSink()
    telemetry = TelemetryProvider(FakeProvider([asyncio.CancelledError()]), sink, clock=SequenceClock(1))

    with pytest.raises(asyncio.CancelledError):
        await telemetry.generate(make_request())

    assert sink.events == []


@pytest.mark.asyncio
async def test_sink_failure_is_best_effort_and_does_not_change_success_or_provider_error() -> None:
    response = make_response()
    success_telemetry = TelemetryProvider(FakeProvider([response]), FailingSink(), clock=SequenceClock(1, 2))

    assert await success_telemetry.generate(make_request()) is response

    provider_error = ProviderUnavailableError("down")
    failure_telemetry = TelemetryProvider(
        FakeProvider([provider_error]),
        FailingSink(),
        clock=SequenceClock(1, 2),
    )
    with pytest.raises(ProviderUnavailableError) as raised:
        await failure_telemetry.generate(make_request())
    assert raised.value is provider_error


@pytest.mark.asyncio
async def test_event_does_not_contain_prompt_messages_or_generated_content() -> None:
    sink = CollectingSink()
    telemetry = TelemetryProvider(FakeProvider([make_response()]), sink, clock=SequenceClock(1, 2))

    await telemetry.generate(make_request())

    event = sink.events[0]
    assert {field.name for field in fields(event)} == {
        "operation",
        "provider",
        "model",
        "duration_ms",
        "success",
        "usage",
        "error_type",
    }
    assert "secret prompt" not in repr(event)
    assert "secret generated content" not in repr(event)


@pytest.mark.asyncio
async def test_fallback_success_records_the_real_winning_provider() -> None:
    primary = FakeProvider([ProviderUnavailableError("down")])
    fallback = FakeProvider([make_response(provider="ollama")])
    sink = CollectingSink()
    telemetry = TelemetryProvider(
        FallbackProvider([primary, fallback]),
        sink,
        clock=SequenceClock(1, 2),
    )

    result = await telemetry.generate(make_request())

    assert result.provider == "ollama"
    assert sink.events[0].provider == "ollama"


@pytest.mark.asyncio
async def test_stream_is_delegated_without_telemetry_events() -> None:
    provider = FakeProvider([make_response()])
    sink = CollectingSink()
    telemetry = TelemetryProvider(provider, sink, clock=SequenceClock())

    chunks = [chunk async for chunk in telemetry.stream(make_request())]

    assert [chunk.content_delta for chunk in chunks] == ["chunk"]
    assert provider.stream_calls == 1
    assert sink.events == []


@pytest.mark.asyncio
async def test_noop_sink_supports_optional_telemetry() -> None:
    response = make_response()
    telemetry = TelemetryProvider(FakeProvider([response]), clock=SequenceClock(1, 2))

    assert await telemetry.generate(make_request()) is response


@pytest.mark.asyncio
async def test_logging_sink_uses_safe_operational_fields_without_global_configuration() -> None:
    logger = FakeLogger()
    sink = LoggingTelemetrySink(logger)
    event = GenerationTelemetryEvent(
        operation="generate",
        provider="openai",
        model="test-model",
        duration_ms=10.0,
        success=True,
        usage=Usage(3, 2),
        error_type=None,
    )

    await sink.emit(event)

    level, _, kwargs = logger.records[0]
    data = kwargs["extra"]["modelcore_telemetry"]
    assert level == "info"
    assert data == {
        "operation": "generate",
        "provider": "openai",
        "model": "test-model",
        "duration_ms": 10.0,
        "success": True,
        "input_tokens": 3,
        "output_tokens": 2,
        "total_tokens": 5,
        "error_type": None,
    }
