import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass

from modelcore.exceptions.provider import (
    GenerationTimeoutError,
    ModelCoreError,
    ProviderUnavailableError,
    RateLimitError,
)
from modelcore.interfaces.llm_provider import LLMProvider
from modelcore.models.chat_request import ChatRequest
from modelcore.models.chat_response import ChatResponse
from modelcore.models.chat_stream_chunk import ChatStreamChunk

Sleep = Callable[[float], Awaitable[None]]
_RETRYABLE_ERRORS = (RateLimitError, ProviderUnavailableError, GenerationTimeoutError)


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Deterministic retry settings for transient generation failures."""

    max_attempts: int = 3
    base_delay: float = 0.5
    max_delay: float | None = None

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.base_delay < 0:
            raise ValueError("base_delay cannot be negative")
        if self.max_delay is not None:
            if self.max_delay < 0:
                raise ValueError("max_delay cannot be negative")
            if self.max_delay < self.base_delay:
                raise ValueError("max_delay cannot be lower than base_delay")

    def delay_for_retry(self, failed_attempt: int) -> float:
        if failed_attempt < 1:
            raise ValueError("failed_attempt must be at least 1")
        delay = self.base_delay * float(2 ** (failed_attempt - 1))
        if self.max_delay is None:
            return delay
        return delay if delay <= self.max_delay else self.max_delay

    def is_retryable(self, error: ModelCoreError) -> bool:
        return isinstance(error, _RETRYABLE_ERRORS)


class ResilientProvider:
    """Adds retry and timeout behavior to a provider through composition."""

    def __init__(
        self,
        provider: LLMProvider,
        retry_policy: RetryPolicy,
        timeout: float | None = None,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be positive")
        self._provider = provider
        self._retry_policy = retry_policy
        self._timeout = timeout
        self._sleep = sleep

    async def generate(self, request: ChatRequest) -> ChatResponse:
        for attempt in range(1, self._retry_policy.max_attempts + 1):
            try:
                return await self._generate_once(request)
            except ModelCoreError as error:
                if not self._retry_policy.is_retryable(error) or attempt == self._retry_policy.max_attempts:
                    raise
                await self._sleep(self._retry_policy.delay_for_retry(attempt))

        raise AssertionError("Retry loop completed without returning or raising")

    def stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        """Delegate streaming unchanged; retrying partial streams is unsafe."""
        return self._provider.stream(request)

    async def _generate_once(self, request: ChatRequest) -> ChatResponse:
        if self._timeout is None:
            return await self._provider.generate(request)
        try:
            async with asyncio.timeout(self._timeout):
                return await self._provider.generate(request)
        except TimeoutError as error:
            raise GenerationTimeoutError("Generation timed out") from error
