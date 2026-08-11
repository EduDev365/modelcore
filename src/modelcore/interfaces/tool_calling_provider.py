from collections.abc import Sequence
from typing import Protocol

from modelcore.models.chat_request import ChatRequest
from modelcore.models.tools import ToolCallingResponse, ToolDefinition, ToolResult


class ToolCallingProvider(Protocol):
    async def generate_with_tools(
        self, request: ChatRequest, tools: Sequence[ToolDefinition]
    ) -> ToolCallingResponse: ...

    async def continue_with_tool_results(
        self,
        request: ChatRequest,
        initial: ToolCallingResponse,
        results: Sequence[ToolResult],
        tools: Sequence[ToolDefinition],
    ) -> ToolCallingResponse: ...
