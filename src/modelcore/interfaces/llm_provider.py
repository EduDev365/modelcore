from collections.abc import AsyncIterator
from typing import Protocol

from modelcore.models.chat_request import ChatRequest
from modelcore.models.chat_response import ChatResponse
from modelcore.models.chat_stream_chunk import ChatStreamChunk


class LLMProvider(Protocol):
    """Generate normalized chat responses and streams."""

    async def generate(self, request: ChatRequest) -> ChatResponse:
        """Generate a normalized response for a chat request."""

    def stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        """Yield normalized incremental updates for a chat request."""
