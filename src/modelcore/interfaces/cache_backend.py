from typing import Protocol

from modelcore.models.chat_response import ChatResponse


class CacheBackend(Protocol):
    """Storage for successful normalized chat responses."""

    async def get(self, key: str) -> ChatResponse | None:
        """Return a non-expired response for a key, if present."""

    async def set(self, key: str, value: ChatResponse, ttl: float | None = None) -> None:
        """Store a response for an optional number of seconds."""
