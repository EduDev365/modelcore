from typing import Any, Protocol

from modelcore.models.chat_request import ChatRequest
from modelcore.models.chat_response import ChatResponse


class StructuredOutputProvider(Protocol):
    """Generate structured JSON content against an application schema."""

    async def generate_structured(
        self,
        request: ChatRequest,
        schema: dict[str, Any],
    ) -> ChatResponse:
        """Generate a normalized JSON response that conforms to a schema."""
