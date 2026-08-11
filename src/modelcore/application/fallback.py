from collections.abc import AsyncIterator, Sequence

from modelcore.exceptions.provider import (
    GenerationTimeoutError,
    ProviderUnavailableError,
    RateLimitError,
)
from modelcore.interfaces.llm_provider import LLMProvider
from modelcore.models.chat_request import ChatRequest
from modelcore.models.chat_response import ChatResponse
from modelcore.models.chat_stream_chunk import ChatStreamChunk

_FALLBACK_ELIGIBLE_ERRORS = (
    GenerationTimeoutError,
    ProviderUnavailableError,
    RateLimitError,
)


class FallbackProvider:
    """Tries chat providers in order after an eligible ModelCore failure."""

    def __init__(self, providers: Sequence[LLMProvider]) -> None:
        normalized_providers = tuple(providers)
        if not normalized_providers:
            raise ValueError("FallbackProvider requires at least one provider")
        if len({id(provider) for provider in normalized_providers}) != len(normalized_providers):
            raise ValueError("FallbackProvider providers must not repeat")
        self._providers = normalized_providers

    async def generate(self, request: ChatRequest) -> ChatResponse:
        for index, provider in enumerate(self._providers):
            try:
                return await provider.generate(request)
            except _FALLBACK_ELIGIBLE_ERRORS:
                if index == len(self._providers) - 1:
                    raise

        raise AssertionError("Fallback provider loop completed without returning or raising")

    def stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        """Delegate only to the primary provider; partial streams cannot safely fallback."""
        return self._providers[0].stream(request)
