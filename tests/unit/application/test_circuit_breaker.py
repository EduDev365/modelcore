import asyncio

import pytest

from modelcore.application import CircuitBreakerPolicy, CircuitBreakerProvider, CircuitState
from modelcore.exceptions import (
    AuthenticationError,
    CircuitOpenError,
    GenerationTimeoutError,
    ProviderError,
    ProviderUnavailableError,
    RateLimitError,
)
from modelcore.models import ChatRequest, ChatResponse, Message


class SequenceClock:
    def __init__(self, *values: float) -> None:
        self.values = iter(values)
        self.current = 0.0

    def __call__(self) -> float:
        self.current = next(self.values)
        return self.current


class FakeProvider:
    def __init__(self, *results: ChatResponse | BaseException) -> None:
        self.results = list(results)
        self.requests: list[ChatRequest] = []

    async def generate(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result

    def stream(self, request):
        return iter(())


def request() -> ChatRequest:
    return ChatRequest([Message.user("hello")], model="test-model")


def response() -> ChatResponse:
    return ChatResponse(content="answer", model="test-model", provider="fake", usage=None)


def test_initial_state_and_policy_validation() -> None:
    provider = FakeProvider(response())
    breaker = CircuitBreakerProvider(provider, CircuitBreakerPolicy())

    assert breaker.state is CircuitState.CLOSED
    assert breaker.consecutive_failures == 0
    assert breaker.opened_at is None
    with pytest.raises(ValueError):
        CircuitBreakerPolicy(failure_threshold=0)
    with pytest.raises(ValueError):
        CircuitBreakerPolicy(recovery_timeout=-1)


@pytest.mark.asyncio
async def test_success_in_closed_returns_exact_response_and_resets_failures() -> None:
    expected = response()
    provider = FakeProvider(RateLimitError("limited"), expected)
    breaker = CircuitBreakerProvider(provider, CircuitBreakerPolicy(failure_threshold=2))

    with pytest.raises(RateLimitError):
        await breaker.generate(request())
    result = await breaker.generate(request())

    assert result is expected
    assert breaker.state is CircuitState.CLOSED
    assert breaker.consecutive_failures == 0


@pytest.mark.asyncio
async def test_threshold_opens_after_consecutive_eligible_failures() -> None:
    provider = FakeProvider(ProviderUnavailableError("down"), RateLimitError("limited"), GenerationTimeoutError("slow"))
    breaker = CircuitBreakerProvider(provider, CircuitBreakerPolicy(failure_threshold=3))

    for expected_state in (CircuitState.CLOSED, CircuitState.CLOSED, CircuitState.OPEN):
        with pytest.raises(Exception):
            await breaker.generate(request())
        assert breaker.state is expected_state
    assert breaker.consecutive_failures == 3


@pytest.mark.asyncio
async def test_open_fails_fast_without_calling_provider() -> None:
    clock = SequenceClock(0.0, 0.0, 1.0)
    provider = FakeProvider(RateLimitError("limited"), response())
    breaker = CircuitBreakerProvider(
        provider, CircuitBreakerPolicy(failure_threshold=1, recovery_timeout=10), clock=clock
    )

    with pytest.raises(RateLimitError):
        await breaker.generate(request())
    with pytest.raises(CircuitOpenError, match="not executed"):
        await breaker.generate(request())

    assert len(provider.requests) == 1


@pytest.mark.asyncio
async def test_half_open_success_closes_and_resets_state() -> None:
    clock = SequenceClock(0.0, 11.0)
    expected = response()
    provider = FakeProvider(RateLimitError("limited"), expected)
    breaker = CircuitBreakerProvider(
        provider, CircuitBreakerPolicy(failure_threshold=1, recovery_timeout=10), clock=clock
    )

    with pytest.raises(RateLimitError):
        await breaker.generate(request())
    result = await breaker.generate(request())

    assert result is expected
    assert breaker.state is CircuitState.CLOSED
    assert breaker.consecutive_failures == 0
    assert breaker.opened_at is None


@pytest.mark.asyncio
async def test_half_open_failure_reopens_from_new_clock_instant() -> None:
    clock = SequenceClock(0.0, 11.0, 11.0)
    provider = FakeProvider(RateLimitError("first"), RateLimitError("again"))
    breaker = CircuitBreakerProvider(
        provider, CircuitBreakerPolicy(failure_threshold=1, recovery_timeout=10), clock=clock
    )

    with pytest.raises(RateLimitError):
        await breaker.generate(request())
    with pytest.raises(RateLimitError):
        await breaker.generate(request())

    assert breaker.state is CircuitState.OPEN
    assert breaker.opened_at == 11.0
    assert len(provider.requests) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [AuthenticationError("auth"), ProviderError("bug")])
async def test_non_eligible_errors_do_not_count_or_open(error: BaseException) -> None:
    provider = FakeProvider(error)
    breaker = CircuitBreakerProvider(provider, CircuitBreakerPolicy(failure_threshold=1))

    with pytest.raises(type(error)):
        await breaker.generate(request())

    assert breaker.state is CircuitState.CLOSED
    assert breaker.consecutive_failures == 0


@pytest.mark.asyncio
async def test_cancelled_error_is_preserved_and_does_not_count() -> None:
    provider = FakeProvider(asyncio.CancelledError())
    breaker = CircuitBreakerProvider(provider, CircuitBreakerPolicy(failure_threshold=1))

    with pytest.raises(asyncio.CancelledError):
        await breaker.generate(request())

    assert breaker.state is CircuitState.CLOSED
    assert breaker.consecutive_failures == 0


@pytest.mark.asyncio
async def test_only_one_half_open_probe_reaches_provider() -> None:
    clock = SequenceClock(10.0)
    started = asyncio.Event()
    release = asyncio.Event()

    class BlockingProvider:
        def __init__(self) -> None:
            self.requests: list[ChatRequest] = []

        async def generate(self, request: ChatRequest) -> ChatResponse:
            self.requests.append(request)
            started.set()
            await release.wait()
            return response()

    provider = BlockingProvider()
    breaker = CircuitBreakerProvider(
        provider, CircuitBreakerPolicy(failure_threshold=1, recovery_timeout=10), clock=clock
    )
    breaker._state = CircuitState.OPEN
    breaker._opened_at = 0.0

    first = asyncio.create_task(breaker.generate(request()))
    await started.wait()
    with pytest.raises(CircuitOpenError):
        await breaker.generate(request())
    release.set()
    await first

    assert len(provider.requests) == 1
    assert breaker.state is CircuitState.CLOSED


@pytest.mark.asyncio
async def test_cancelled_half_open_probe_does_not_leave_half_open() -> None:
    clock = SequenceClock(10.0, 10.0, 10.0)
    started = asyncio.Event()

    class CancelProvider:
        async def generate(self, request: ChatRequest) -> ChatResponse:
            started.set()
            await asyncio.Event().wait()
            return response()

    breaker = CircuitBreakerProvider(
        CancelProvider(), CircuitBreakerPolicy(failure_threshold=1, recovery_timeout=10), clock=clock
    )
    breaker._state = CircuitState.OPEN
    breaker._opened_at = 0.0
    task = asyncio.create_task(breaker.generate(request()))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert breaker.state is CircuitState.OPEN
    with pytest.raises(CircuitOpenError):
        await breaker.generate(request())
