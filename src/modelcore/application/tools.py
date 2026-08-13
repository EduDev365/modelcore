import asyncio
import inspect
import json
from collections.abc import Sequence

from pydantic import ValidationError

from modelcore.exceptions.tool import ToolExecutionError, ToolNotFoundError, ToolRoundLimitError, ToolValidationError
from modelcore.interfaces.tool_calling_provider import ToolCallingProvider
from modelcore.models.chat_request import ChatRequest
from modelcore.models.tools import ToolCall, ToolCallingResponse, ToolDefinition, ToolResult


class ToolRegistry:
    """Registry of explicitly allowlisted tool definitions keyed by name."""

    def __init__(self, tools: Sequence[ToolDefinition] = ()) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: ToolDefinition) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolDefinition:
        try:
            return self._tools[name]
        except KeyError as error:
            raise ToolNotFoundError("Requested tool is not registered") from error


class ToolExecutor:
    """Validate one tool call and invoke its registered sync or async handler."""

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    async def execute(self, call: ToolCall) -> ToolResult:
        tool = self._registry.get(call.name)
        try:
            arguments = tool.arguments_model.model_validate(call.arguments).model_dump()
        except ValidationError as error:
            raise ToolValidationError("Tool arguments did not match the registered schema") from error
        try:
            value = tool.handler(**arguments)
            if inspect.isawaitable(value):
                value = await value
            content = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        except asyncio.CancelledError:
            raise
        except Exception as error:
            raise ToolExecutionError("Registered tool execution failed") from error
        return ToolResult(call.id, tool.name, content)


class ToolGeneration:
    """Run one bounded tool-calling round through a compatible provider."""

    def __init__(self, provider: ToolCallingProvider, executor: ToolExecutor) -> None:
        self._provider = provider
        self._executor = executor

    async def generate(self, request: ChatRequest, tools: Sequence[ToolDefinition]) -> ToolCallingResponse:
        normalized_tools = tuple(tools)
        initial = await self._provider.generate_with_tools(request, normalized_tools)
        if not initial.tool_calls:
            return initial
        results = tuple([await self._executor.execute(call) for call in initial.tool_calls])
        final = await self._provider.continue_with_tool_results(request, initial, results, normalized_tools)
        if final.tool_calls:
            raise ToolRoundLimitError("Tool calling is limited to one execution round")
        return final
