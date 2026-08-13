"""Circuit breaker wrapper for normal, non-streaming generation."""

import asyncio
import math
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from enum import Enum

from modelcore.exceptions.provider import (
    CircuitOpenError,
    GenerationTimeoutError,
    ProviderUnavailableError,
    RateLimitError,
)
from modelcore.interfaces.llm_provider import LLMProvider
from modelcore.models.chat_request import ChatRequest
from modelcore.models.chat_response import ChatResponse
from modelcore.models.chat_stream_chunk import ChatStreamChunk

CircuitClock = Callable[[], float]
_ELIGIBLE_FAILURES = (RateLimitError, ProviderUnavailableError, GenerationTimeoutError)


class CircuitState(Enum):
    """Operational state of a circuit breaker."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass(frozen=True, slots=True)
class CircuitBreakerPolicy:
    """Immutable thresholds controlling circuit opening and recovery tests."""

    failure_threshold: int = 5
    recovery_timeout: float = 30.0

    def __post_init__(self) -> None:
        if isinstance(self.failure_threshold, bool) or not isinstance(self.failure_threshold, int):
            raise ValueError("failure_threshold must be an integer at least 1")
        if self.failure_threshold < 1:
            raise ValueError("failure_threshold must be at least 1")
        if isinstance(self.recovery_timeout, bool) or not isinstance(self.recovery_timeout, (int, float)):
            raise ValueError("recovery_timeout must be a finite non-negative number")
        if not math.isfinite(self.recovery_timeout) or self.recovery_timeout < 0:
            raise ValueError("recovery_timeout must be a finite non-negative number")
        object.__setattr__(self, "recovery_timeout", float(self.recovery_timeout))


class CircuitBreakerProvider:
    """Protect a provider from repeated transient generation failures."""

    def __init__(
        self,
        provider: LLMProvider,
        policy: CircuitBreakerPolicy,
        *,
        clock: CircuitClock = time.monotonic,
    ) -> None:
        self._provider = provider
        self._policy = policy
        self._clock = clock
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._opened_at: float | None = None
        self._half_open_in_flight = False
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        return self._state

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    @property
    def opened_at(self) -> float | None:
        return self._opened_at

    async def generate(self, request: ChatRequest) -> ChatResponse:
        half_open_attempt = await self._before_call()
        try:
            response = await self._provider.generate(request)
        except asyncio.CancelledError:
            await self._handle_cancelled(half_open_attempt)
            raise
        except _ELIGIBLE_FAILURES:
            await self._handle_eligible_failure(half_open_attempt)
            raise
        except Exception:
            await self._handle_non_eligible_failure(half_open_attempt)
            raise
        else:
            await self._handle_success(half_open_attempt)
            return response

    def stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        """Delegate streaming unchanged; circuit breaking is generate-only."""
        return self._provider.stream(request)

    async def _before_call(self) -> bool:
        async with self._lock:
            if self._state is CircuitState.CLOSED:
                return False
            if self._state is CircuitState.HALF_OPEN:
                raise CircuitOpenError()
            assert self._opened_at is not None
            if self._clock() - self._opened_at < self._policy.recovery_timeout:
                raise CircuitOpenError()
            self._state = CircuitState.HALF_OPEN
            self._half_open_in_flight = True
            return True

    async def _handle_success(self, half_open_attempt: bool) -> None:
        async with self._lock:
            if half_open_attempt:
                self._close()
            elif self._state is CircuitState.CLOSED:
                self._consecutive_failures = 0

    async def _handle_eligible_failure(self, half_open_attempt: bool) -> None:
        async with self._lock:
            if half_open_attempt:
                self._open()
                return
            if self._state is not CircuitState.CLOSED:
                return
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._policy.failure_threshold:
                self._open()

    async def _handle_non_eligible_failure(self, half_open_attempt: bool) -> None:
        async with self._lock:
            if half_open_attempt:
                self._close()

    async def _handle_cancelled(self, half_open_attempt: bool) -> None:
        if not half_open_attempt:
            return
        async with self._lock:
            if self._state is CircuitState.HALF_OPEN:
                self._open()

    def _open(self) -> None:
        self._state = CircuitState.OPEN
        self._opened_at = self._clock()
        self._half_open_in_flight = False

    def _close(self) -> None:
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._opened_at = None
        self._half_open_in_flight = False
