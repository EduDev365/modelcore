import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass

from modelcore.exceptions.provider import (
    GenerationTimeoutError,
    ModelCoreError,
    ProviderUnavailableError,
    RateLimitError,
)
from modelcore.interfaces.llm_provider import LLMProvider
from modelcore.interfaces.retry_telemetry_sink import RetryTelemetrySink
from modelcore.models.chat_request import ChatRequest
from modelcore.models.chat_response import ChatResponse
from modelcore.models.chat_stream_chunk import ChatStreamChunk
from modelcore.models.retry_telemetry import RetryTelemetryEvent

Sleep = Callable[[float], Awaitable[None]]
RetryClock = Callable[[], float]
_RETRYABLE_ERRORS = (RateLimitError, ProviderUnavailableError, GenerationTimeoutError)


class NoOpRetryTelemetrySink:
    """Retry telemetry sink that intentionally discards events."""

    async def emit(self, event: RetryTelemetryEvent) -> None:
        return None


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
        *,
        provider_name: str | None = None,
        telemetry_sink: RetryTelemetrySink | None = None,
        clock: RetryClock = time.monotonic,
    ) -> None:
        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be positive")
        if provider_name is not None and not provider_name.strip():
            raise ValueError("provider_name cannot be blank")
        if telemetry_sink is not None and provider_name is None:
            raise ValueError("provider_name is required when telemetry_sink is configured")
        self._provider = provider
        self._retry_policy = retry_policy
        self._timeout = timeout
        self._sleep = sleep
        self._provider_name = provider_name
        self._telemetry_sink = telemetry_sink
        self._clock = clock

    async def generate(self, request: ChatRequest) -> ChatResponse:
        for attempt in range(1, self._retry_policy.max_attempts + 1):
            started_at = self._clock() if self._telemetry_sink is not None else 0.0
            try:
                response = await self._generate_once(request)
            except ModelCoreError as error:
                retryable = self._retry_policy.is_retryable(error)
                if not retryable or attempt == self._retry_policy.max_attempts:
                    outcome = "exhausted" if retryable else "error"
                    await self._emit_best_effort(
                        self._event(request, attempt, outcome, started_at, error_type=type(error).__name__)
                    )
                    raise
                delay = self._retry_policy.delay_for_retry(attempt)
                await self._emit_best_effort(
                    self._event(
                        request,
                        attempt,
                        "retry",
                        started_at,
                        delay_ms=delay * 1000,
                        error_type=type(error).__name__,
                    )
                )
                await self._sleep(delay)
            else:
                await self._emit_best_effort(self._event(request, attempt, "success", started_at))
                return response

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

    def _event(
        self,
        request: ChatRequest,
        attempt: int,
        outcome: str,
        started_at: float,
        *,
        delay_ms: float | None = None,
        error_type: str | None = None,
    ) -> RetryTelemetryEvent | None:
        if self._telemetry_sink is None:
            return None
        if self._provider_name is None:
            raise AssertionError("provider_name is required for retry telemetry")
        return RetryTelemetryEvent(
            provider=self._provider_name,
            model=request.model,
            attempt=attempt,
            max_attempts=self._retry_policy.max_attempts,
            outcome=outcome,
            duration_ms=(self._clock() - started_at) * 1000,
            delay_ms=delay_ms,
            error_type=error_type,
        )

    async def _emit_best_effort(self, event: RetryTelemetryEvent | None) -> None:
        if self._telemetry_sink is None or event is None:
            return
        try:
            await self._telemetry_sink.emit(event)
        except Exception:
            pass
