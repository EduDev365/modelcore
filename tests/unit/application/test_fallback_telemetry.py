import asyncio
from collections.abc import AsyncIterator

import pytest

from modelcore.application import FallbackProvider, ResilientProvider, RetryPolicy
from modelcore.exceptions import (
    AuthenticationError,
    GenerationTimeoutError,
    ProviderError,
    ProviderUnavailableError,
    RateLimitError,
)
from modelcore.models import (
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    FallbackTelemetryEvent,
    Message,
    RetryTelemetryEvent,
)


class SequenceClock:
    def __init__(self, *values: float) -> None:
        self._values = iter(values)

    def __call__(self) -> float:
        return next(self._values)


class FallbackSink:
    def __init__(self) -> None:
        self.events: list[FallbackTelemetryEvent] = []

    async def emit(self, event: FallbackTelemetryEvent) -> None:
        self.events.append(event)


class RetrySink:
    def __init__(self) -> None:
        self.events: list[RetryTelemetryEvent] = []

    async def emit(self, event: RetryTelemetryEvent) -> None:
        self.events.append(event)


class FailingSink:
    async def emit(self, event: FallbackTelemetryEvent) -> None:
        raise RuntimeError("sink secret")


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


def request() -> ChatRequest:
    return ChatRequest([Message.user("secret prompt")], model="test-model")


def response(provider: str = "actual-provider") -> ChatResponse:
    return ChatResponse("secret response", "test-model", provider, None)


@pytest.mark.asyncio
async def test_primary_success_uses_configured_candidate_identity_not_response_provider() -> None:
    sink = FallbackSink()
    fallback = FallbackProvider(
        [FakeProvider([response("sdk-normalized-provider")]), FakeProvider([response()])],
        provider_names=["logical-primary", "logical-secondary"],
        telemetry_sink=sink,
        clock=SequenceClock(1, 1.5),
    )

    result = await fallback.generate(request())

    assert result.provider == "sdk-normalized-provider"
    assert sink.events == [FallbackTelemetryEvent("logical-primary", "test-model", 1, 2, "success", 500.0)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [RateLimitError("limited"), ProviderUnavailableError("down"), GenerationTimeoutError("slow")],
)
async def test_eligible_failure_emits_fallback_then_secondary_success(error: ProviderError) -> None:
    sink = FallbackSink()
    first = FakeProvider([error])
    second = FakeProvider([response("actual-secondary")])
    fallback = FallbackProvider(
        [first, second],
        provider_names=["primary", "secondary"],
        telemetry_sink=sink,
        clock=SequenceClock(1, 1.1, 2, 2.2),
    )

    await fallback.generate(request())

    assert [(event.provider, event.candidate_index, event.outcome) for event in sink.events] == [
        ("primary", 1, "fallback"),
        ("secondary", 2, "success"),
    ]
    assert sink.events[0].error_type == type(error).__name__


@pytest.mark.asyncio
async def test_multiple_candidates_exhaust_and_preserve_last_exception() -> None:
    last = GenerationTimeoutError("last secret")
    sink = FallbackSink()
    fallback = FallbackProvider(
        [
            FakeProvider([ProviderUnavailableError("first")]),
            FakeProvider([RateLimitError("second")]),
            FakeProvider([last]),
        ],
        provider_names=["one", "two", "three"],
        telemetry_sink=sink,
        clock=SequenceClock(1, 2, 3, 4, 5, 6),
    )

    with pytest.raises(GenerationTimeoutError) as caught:
        await fallback.generate(request())

    assert caught.value is last
    assert [(event.provider, event.outcome) for event in sink.events] == [
        ("one", "fallback"),
        ("two", "fallback"),
        ("three", "exhausted"),
    ]
    assert "secret" not in repr(sink.events)


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [AuthenticationError("bad key"), ProviderError("bad request")])
async def test_non_eligible_error_emits_error_and_stops_chain(error: ProviderError) -> None:
    sink = FallbackSink()
    secondary = FakeProvider([response()])
    fallback = FallbackProvider(
        [FakeProvider([error]), secondary],
        provider_names=["primary", "secondary"],
        telemetry_sink=sink,
        clock=SequenceClock(1, 2),
    )

    with pytest.raises(type(error)) as caught:
        await fallback.generate(request())

    assert caught.value is error
    assert secondary.calls == 0
    assert sink.events[0].outcome == "error"


@pytest.mark.asyncio
async def test_sink_failure_does_not_change_success_or_original_failure() -> None:
    successful = FallbackProvider(
        [FakeProvider([ProviderUnavailableError("down")]), FakeProvider([response()])],
        provider_names=["one", "two"],
        telemetry_sink=FailingSink(),
    )
    assert await successful.generate(request()) == response()

    error = AuthenticationError("original")
    failing = FallbackProvider(
        [FakeProvider([error]), FakeProvider([response()])],
        provider_names=["one", "two"],
        telemetry_sink=FailingSink(),
    )
    with pytest.raises(AuthenticationError) as caught:
        await failing.generate(request())
    assert caught.value is error


@pytest.mark.asyncio
async def test_cancellation_propagates_without_fallback_event_or_next_candidate() -> None:
    sink = FallbackSink()
    secondary = FakeProvider([response()])
    fallback = FallbackProvider(
        [FakeProvider([asyncio.CancelledError()]), secondary],
        provider_names=["one", "two"],
        telemetry_sink=sink,
        clock=SequenceClock(1),
    )

    with pytest.raises(asyncio.CancelledError):
        await fallback.generate(request())
    assert sink.events == []
    assert secondary.calls == 0


@pytest.mark.asyncio
async def test_retry_and_fallback_events_remain_separate_in_composition() -> None:
    retry_sink = RetrySink()
    fallback_sink = FallbackSink()
    primary = ResilientProvider(
        FakeProvider([ProviderUnavailableError("first"), ProviderUnavailableError("last")]),
        RetryPolicy(max_attempts=2, base_delay=0),
        sleep=lambda _: asyncio.sleep(0),
        provider_name="openai",
        telemetry_sink=retry_sink,
    )
    secondary = ResilientProvider(
        FakeProvider([response("ollama-runtime")]),
        RetryPolicy(max_attempts=2),
        provider_name="ollama",
        telemetry_sink=retry_sink,
    )
    fallback = FallbackProvider(
        [primary, secondary],
        provider_names=["openai", "ollama"],
        telemetry_sink=fallback_sink,
    )

    result = await fallback.generate(request())

    assert result.provider == "ollama-runtime"
    assert [(event.provider, event.attempt, event.outcome) for event in retry_sink.events] == [
        ("openai", 1, "retry"),
        ("openai", 2, "exhausted"),
        ("ollama", 1, "success"),
    ]
    assert [(event.provider, event.outcome) for event in fallback_sink.events] == [
        ("openai", "fallback"),
        ("ollama", "success"),
    ]


@pytest.mark.asyncio
async def test_all_resilient_candidates_exhaust_and_preserve_last_error() -> None:
    retry_sink = RetrySink()
    fallback_sink = FallbackSink()
    final_error = GenerationTimeoutError("final")
    providers = [
        ResilientProvider(
            FakeProvider([RateLimitError("first")]),
            RetryPolicy(max_attempts=1),
            provider_name="openai",
            telemetry_sink=retry_sink,
        ),
        ResilientProvider(
            FakeProvider([final_error]),
            RetryPolicy(max_attempts=1),
            provider_name="ollama",
            telemetry_sink=retry_sink,
        ),
    ]
    fallback = FallbackProvider(
        providers,
        provider_names=["openai", "ollama"],
        telemetry_sink=fallback_sink,
    )

    with pytest.raises(GenerationTimeoutError) as caught:
        await fallback.generate(request())

    assert caught.value is final_error
    assert [(event.provider, event.outcome) for event in retry_sink.events] == [
        ("openai", "exhausted"),
        ("ollama", "exhausted"),
    ]
    assert [(event.provider, event.outcome) for event in fallback_sink.events] == [
        ("openai", "fallback"),
        ("ollama", "exhausted"),
    ]


@pytest.mark.asyncio
async def test_authentication_error_in_resilient_primary_neither_retries_nor_falls_back() -> None:
    retry_sink = RetrySink()
    fallback_sink = FallbackSink()
    error = AuthenticationError("bad key")
    secondary_provider = FakeProvider([response()])
    providers = [
        ResilientProvider(
            FakeProvider([error]),
            RetryPolicy(max_attempts=3),
            provider_name="openai",
            telemetry_sink=retry_sink,
        ),
        ResilientProvider(
            secondary_provider,
            RetryPolicy(max_attempts=3),
            provider_name="ollama",
            telemetry_sink=retry_sink,
        ),
    ]
    fallback = FallbackProvider(
        providers,
        provider_names=["openai", "ollama"],
        telemetry_sink=fallback_sink,
    )

    with pytest.raises(AuthenticationError) as caught:
        await fallback.generate(request())

    assert caught.value is error
    assert secondary_provider.calls == 0
    assert [(event.provider, event.attempt, event.outcome) for event in retry_sink.events] == [("openai", 1, "error")]
    assert [(event.provider, event.outcome) for event in fallback_sink.events] == [("openai", "error")]


def test_fallback_telemetry_identity_validation_is_additive_and_strict() -> None:
    providers = [FakeProvider([response()]), FakeProvider([response()])]

    FallbackProvider(providers)
    FallbackProvider(providers, provider_names=["one", "two"])
    with pytest.raises(ValueError, match="provider_names is required"):
        FallbackProvider(providers, telemetry_sink=FallbackSink())
    with pytest.raises(ValueError, match="match providers length"):
        FallbackProvider(providers, provider_names=["one"])
    with pytest.raises(ValueError, match="blank"):
        FallbackProvider(providers, provider_names=["one", " "])
    with pytest.raises(ValueError, match="unique"):
        FallbackProvider(providers, provider_names=["same", "same"])
