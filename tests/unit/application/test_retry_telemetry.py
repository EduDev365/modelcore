import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable

import pytest

from modelcore.application import ResilientProvider, RetryPolicy
from modelcore.exceptions import (
    AuthenticationError,
    GenerationTimeoutError,
    ProviderError,
    ProviderUnavailableError,
    RateLimitError,
)
from modelcore.models import ChatRequest, ChatResponse, ChatStreamChunk, Message, RetryTelemetryEvent


class SequenceClock:
    def __init__(self, *values: float) -> None:
        self._values = iter(values)

    def __call__(self) -> float:
        return next(self._values)


class CollectingSink:
    def __init__(self) -> None:
        self.events: list[RetryTelemetryEvent] = []

    async def emit(self, event: RetryTelemetryEvent) -> None:
        self.events.append(event)


class FailingSink:
    async def emit(self, event: RetryTelemetryEvent) -> None:
        raise RuntimeError("telemetry unavailable with api_key=secret")


class FakeProvider:
    def __init__(self, outcomes: list[ChatResponse | BaseException]) -> None:
        self._outcomes = outcomes
        self.calls = 0

    async def generate(self, request: ChatRequest) -> ChatResponse:
        outcome = self._outcomes[self.calls]
        self.calls += 1
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        async def empty() -> AsyncIterator[ChatStreamChunk]:
            if False:
                yield

        return empty()


def make_request() -> ChatRequest:
    return ChatRequest([Message.user("secret prompt")], model="test-model")


def make_response() -> ChatResponse:
    return ChatResponse("secret generated content", "test-model", "actual-provider", None)


def recording_sleep(delays: list[float]) -> Callable[[float], Awaitable[None]]:
    async def sleep(delay: float) -> None:
        delays.append(delay)

    return sleep


@pytest.mark.asyncio
async def test_first_attempt_success_emits_configured_identity_and_duration() -> None:
    sink = CollectingSink()
    resilient = ResilientProvider(
        FakeProvider([make_response()]),
        RetryPolicy(max_attempts=3),
        provider_name="logical-primary",
        telemetry_sink=sink,
        clock=SequenceClock(1, 1.25),
    )

    response = await resilient.generate(make_request())

    assert response.provider == "actual-provider"
    assert sink.events == [RetryTelemetryEvent("logical-primary", "test-model", 1, 3, "success", 250.0)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [RateLimitError("limited"), ProviderUnavailableError("down"), GenerationTimeoutError("slow")],
)
async def test_transient_error_emits_retry_with_policy_delay_then_success(error: ProviderError) -> None:
    sink = CollectingSink()
    delays: list[float] = []
    resilient = ResilientProvider(
        FakeProvider([error, make_response()]),
        RetryPolicy(max_attempts=3, base_delay=0.25),
        sleep=recording_sleep(delays),
        provider_name="openai",
        telemetry_sink=sink,
        clock=SequenceClock(1, 1.1, 2, 2.2),
    )

    await resilient.generate(make_request())

    assert delays == [0.25]
    assert sink.events[0] == RetryTelemetryEvent(
        "openai", "test-model", 1, 3, "retry", pytest.approx(100), 250.0, type(error).__name__
    )
    assert sink.events[1] == RetryTelemetryEvent("openai", "test-model", 2, 3, "success", pytest.approx(200))


@pytest.mark.asyncio
async def test_retry_exhaustion_is_distinct_and_preserves_last_error() -> None:
    first = ProviderUnavailableError("first secret")
    last = ProviderUnavailableError("last secret")
    sink = CollectingSink()
    resilient = ResilientProvider(
        FakeProvider([first, last]),
        RetryPolicy(max_attempts=2, base_delay=0),
        sleep=recording_sleep([]),
        provider_name="openai",
        telemetry_sink=sink,
        clock=SequenceClock(1, 2, 3, 4),
    )

    with pytest.raises(ProviderUnavailableError) as caught:
        await resilient.generate(make_request())

    assert caught.value is last
    assert [(event.attempt, event.outcome, event.error_type) for event in sink.events] == [
        (1, "retry", "ProviderUnavailableError"),
        (2, "exhausted", "ProviderUnavailableError"),
    ]
    assert "secret" not in repr(sink.events)


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [AuthenticationError("bad key"), ProviderError("bad request")])
async def test_non_retryable_error_is_not_exhaustion(error: ProviderError) -> None:
    sink = CollectingSink()
    provider = FakeProvider([error])
    resilient = ResilientProvider(
        provider,
        RetryPolicy(),
        provider_name="openai",
        telemetry_sink=sink,
        clock=SequenceClock(1, 2),
    )

    with pytest.raises(type(error)) as caught:
        await resilient.generate(make_request())

    assert caught.value is error
    assert provider.calls == 1
    assert sink.events[0].outcome == "error"
    assert sink.events[0].error_type == type(error).__name__


@pytest.mark.asyncio
async def test_sink_failure_never_changes_retry_success_or_provider_error() -> None:
    delays: list[float] = []
    successful = ResilientProvider(
        FakeProvider([RateLimitError("limited"), make_response()]),
        RetryPolicy(max_attempts=2, base_delay=0),
        sleep=recording_sleep(delays),
        provider_name="openai",
        telemetry_sink=FailingSink(),
    )
    assert await successful.generate(make_request()) == make_response()
    assert delays == [0]

    error = AuthenticationError("original")
    failing = ResilientProvider(
        FakeProvider([error]),
        RetryPolicy(),
        provider_name="openai",
        telemetry_sink=FailingSink(),
    )
    with pytest.raises(AuthenticationError) as caught:
        await failing.generate(make_request())
    assert caught.value is error


@pytest.mark.asyncio
async def test_cancellation_propagates_without_event_or_sleep() -> None:
    sink = CollectingSink()
    delays: list[float] = []
    resilient = ResilientProvider(
        FakeProvider([asyncio.CancelledError()]),
        RetryPolicy(),
        sleep=recording_sleep(delays),
        provider_name="openai",
        telemetry_sink=sink,
        clock=SequenceClock(1),
    )

    with pytest.raises(asyncio.CancelledError):
        await resilient.generate(make_request())
    assert sink.events == []
    assert delays == []


def test_retry_telemetry_identity_validation_is_additive() -> None:
    provider = FakeProvider([make_response()])
    policy = RetryPolicy()

    ResilientProvider(provider, policy)
    ResilientProvider(provider, policy, provider_name="openai")
    with pytest.raises(ValueError, match="provider_name cannot be blank"):
        ResilientProvider(provider, policy, provider_name=" ")
    with pytest.raises(ValueError, match="provider_name is required"):
        ResilientProvider(provider, policy, telemetry_sink=CollectingSink())
