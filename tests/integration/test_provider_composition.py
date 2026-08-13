"""Deterministic integration tests for explicit provider wrapper composition."""

import asyncio
from collections.abc import AsyncIterator

import pytest

from modelcore.application import (
    CachingProvider,
    CheapPolicy,
    CircuitBreakerPolicy,
    CircuitBreakerProvider,
    CircuitState,
    FallbackProvider,
    MemoryCache,
    ModelCandidate,
    ResilientProvider,
    RetryPolicy,
    RoutingProvider,
    TelemetryProvider,
)
from modelcore.application.cache import build_cache_key
from modelcore.exceptions import (
    AuthenticationError,
    CircuitOpenError,
    GenerationTimeoutError,
    ProviderUnavailableError,
    RateLimitError,
)
from modelcore.models import ChatRequest, ChatResponse, ChatStreamChunk, GenerationTelemetryEvent, Message
from modelcore.models.routing_telemetry import RoutingTelemetryEvent


class Clock:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


class FakeProvider:
    def __init__(self, *outcomes: ChatResponse | BaseException, name: str = "fake") -> None:
        self.outcomes = list(outcomes)
        self.name = name
        self.generate_calls = 0
        self.stream_calls = 0

    async def generate(self, request: ChatRequest) -> ChatResponse:
        self.generate_calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    async def _stream(self) -> AsyncIterator[ChatStreamChunk]:
        yield ChatStreamChunk("chunk", "test-model", self.name)

    def stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        self.stream_calls += 1
        return self._stream()


class CollectingGenerationSink:
    def __init__(self, *, fail: bool = False) -> None:
        self.events: list[GenerationTelemetryEvent] = []
        self.fail = fail

    async def emit(self, event: GenerationTelemetryEvent) -> None:
        self.events.append(event)
        if self.fail:
            raise RuntimeError("sink unavailable")


class CollectingRoutingSink:
    def __init__(self, *, fail: bool = False) -> None:
        self.events: list[RoutingTelemetryEvent] = []
        self.fail = fail

    async def emit(self, event: RoutingTelemetryEvent) -> None:
        self.events.append(event)
        if self.fail:
            raise RuntimeError("sink unavailable")


async def no_sleep(delay: float) -> None:
    return None


def request() -> ChatRequest:
    return ChatRequest([Message.user("integration")], model="test-model", temperature=0.2)


def response(provider: str = "fake") -> ChatResponse:
    return ChatResponse("answer", "test-model", provider, None)


def resilient(provider: FakeProvider | CircuitBreakerProvider, *, attempts: int = 2) -> ResilientProvider:
    return ResilientProvider(
        provider,
        RetryPolicy(max_attempts=attempts, base_delay=0),
        sleep=no_sleep,
    )


@pytest.mark.asyncio
async def test_retry_recovery_is_one_success_observed_by_outer_breaker() -> None:
    inner = FakeProvider(ProviderUnavailableError("temporary"), response())
    breaker = CircuitBreakerProvider(resilient(inner), CircuitBreakerPolicy(failure_threshold=1))

    result = await breaker.generate(request())

    assert result.provider == "fake"
    assert inner.generate_calls == 2
    assert breaker.state is CircuitState.CLOSED
    assert breaker.consecutive_failures == 0


@pytest.mark.asyncio
async def test_retry_exhaustion_counts_once_in_outer_breaker() -> None:
    inner = FakeProvider(*(ProviderUnavailableError("down") for _ in range(3)))
    breaker = CircuitBreakerProvider(
        resilient(inner, attempts=3),
        CircuitBreakerPolicy(failure_threshold=2),
    )

    with pytest.raises(ProviderUnavailableError):
        await breaker.generate(request())

    assert inner.generate_calls == 3
    assert breaker.state is CircuitState.CLOSED
    assert breaker.consecutive_failures == 1


@pytest.mark.asyncio
async def test_open_circuit_triggers_fallback_without_reaching_primary() -> None:
    clock = Clock()
    primary_inner = FakeProvider(ProviderUnavailableError("down"))
    primary = CircuitBreakerProvider(
        primary_inner,
        CircuitBreakerPolicy(failure_threshold=1, recovery_timeout=30),
        clock=clock,
    )
    secondary = FakeProvider(response("secondary"))
    with pytest.raises(ProviderUnavailableError):
        await primary.generate(request())

    result = await FallbackProvider([primary, secondary]).generate(request())

    assert result.provider == "secondary"
    assert primary_inner.generate_calls == 1
    assert secondary.generate_calls == 1


@pytest.mark.asyncio
async def test_open_circuit_does_not_trigger_retry() -> None:
    clock = Clock()
    inner = FakeProvider(ProviderUnavailableError("down"))
    breaker = CircuitBreakerProvider(
        inner,
        CircuitBreakerPolicy(failure_threshold=1, recovery_timeout=30),
        clock=clock,
    )
    with pytest.raises(ProviderUnavailableError):
        await breaker.generate(request())

    with pytest.raises(CircuitOpenError):
        await resilient(breaker, attempts=3).generate(request())

    assert inner.generate_calls == 1


@pytest.mark.asyncio
async def test_last_circuit_open_error_is_preserved_when_all_fallbacks_are_open() -> None:
    clock = Clock()
    breakers: list[CircuitBreakerProvider] = []
    inners: list[FakeProvider] = []
    for _ in range(2):
        inner = FakeProvider(RateLimitError("down"))
        breaker = CircuitBreakerProvider(
            inner,
            CircuitBreakerPolicy(failure_threshold=1, recovery_timeout=30),
            clock=clock,
        )
        with pytest.raises(RateLimitError):
            await breaker.generate(request())
        inners.append(inner)
        breakers.append(breaker)

    with pytest.raises(CircuitOpenError, match="not executed"):
        await FallbackProvider(breakers).generate(request())

    assert [inner.generate_calls for inner in inners] == [1, 1]


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [RateLimitError, GenerationTimeoutError, ProviderUnavailableError])
async def test_transient_retry_exhaustion_counts_once_then_falls_back(error_type: type[Exception]) -> None:
    primary_inner = FakeProvider(error_type("first"), error_type("second"))
    primary = CircuitBreakerProvider(
        resilient(primary_inner),
        CircuitBreakerPolicy(failure_threshold=2),
    )
    secondary = FakeProvider(response("secondary"))

    result = await FallbackProvider([primary, secondary]).generate(request())

    assert result.provider == "secondary"
    assert primary_inner.generate_calls == 2
    assert primary.consecutive_failures == 1
    assert secondary.generate_calls == 1


@pytest.mark.asyncio
async def test_authentication_error_is_not_retried_counted_or_fallen_back() -> None:
    primary_inner = FakeProvider(AuthenticationError("invalid credentials"))
    primary = CircuitBreakerProvider(
        resilient(primary_inner, attempts=3),
        CircuitBreakerPolicy(failure_threshold=1),
    )
    secondary = FakeProvider(response("secondary"))

    with pytest.raises(AuthenticationError):
        await FallbackProvider([primary, secondary]).generate(request())

    assert primary_inner.generate_calls == 1
    assert primary.state is CircuitState.CLOSED
    assert primary.consecutive_failures == 0
    assert secondary.generate_calls == 0


@pytest.mark.asyncio
async def test_cache_hit_bypasses_fallback_retry_and_breaker() -> None:
    cache = MemoryCache()
    cached_response = response("cached")
    await cache.set(build_cache_key("composed", request()), cached_response)
    primary_inner = FakeProvider(ProviderUnavailableError("must not run"))
    primary = CircuitBreakerProvider(resilient(primary_inner), CircuitBreakerPolicy(failure_threshold=1))
    secondary = FakeProvider(response("secondary"))
    provider = CachingProvider(FallbackProvider([primary, secondary]), cache, provider_key="composed")

    result = await provider.generate(request())

    assert result is cached_response
    assert primary_inner.generate_calls == secondary.generate_calls == 0
    assert primary.state is CircuitState.CLOSED


@pytest.mark.asyncio
async def test_cache_miss_executes_lower_layers_then_stores_result() -> None:
    cache = MemoryCache()
    primary = FakeProvider(response("primary"))
    secondary = FakeProvider(response("secondary"))
    provider = CachingProvider(FallbackProvider([primary, secondary]), cache, provider_key="composed")

    first = await provider.generate(request())
    second = await provider.generate(request())

    assert first is second
    assert primary.generate_calls == 1
    assert secondary.generate_calls == 0


@pytest.mark.asyncio
async def test_telemetry_position_controls_cache_hit_observation() -> None:
    outer_sink = CollectingGenerationSink()
    outer_inner = FakeProvider(response())
    outer = TelemetryProvider(
        CachingProvider(outer_inner, MemoryCache(), provider_key="outer"),
        outer_sink,
        clock=Clock(),
    )
    inner_sink = CollectingGenerationSink()
    inner_provider = FakeProvider(response())
    inner = CachingProvider(
        TelemetryProvider(inner_provider, inner_sink, clock=Clock()),
        MemoryCache(),
        provider_key="inner",
    )

    await outer.generate(request())
    await outer.generate(request())
    await inner.generate(request())
    await inner.generate(request())

    assert len(outer_sink.events) == 2
    assert len(inner_sink.events) == 1
    assert outer_inner.generate_calls == inner_provider.generate_calls == 1


@pytest.mark.asyncio
async def test_telemetry_sink_failures_do_not_change_composed_response_or_error() -> None:
    successful = FakeProvider(response("secondary"))
    observed_success = TelemetryProvider(successful, CollectingGenerationSink(fail=True), clock=Clock())
    assert (await observed_success.generate(request())).provider == "secondary"

    original = ProviderUnavailableError("original")
    failing = TelemetryProvider(FakeProvider(original), CollectingGenerationSink(fail=True), clock=Clock())
    with pytest.raises(ProviderUnavailableError) as raised:
        await failing.generate(request())
    assert raised.value is original


@pytest.mark.asyncio
async def test_cancelled_error_crosses_deep_composition_without_state_corruption() -> None:
    primary_inner = FakeProvider(asyncio.CancelledError())
    breaker = CircuitBreakerProvider(resilient(primary_inner), CircuitBreakerPolicy(failure_threshold=1))
    secondary = FakeProvider(response("secondary"))
    composed = TelemetryProvider(
        CachingProvider(FallbackProvider([breaker, secondary]), MemoryCache(), provider_key="deep"),
        CollectingGenerationSink(fail=True),
        clock=Clock(),
    )

    with pytest.raises(asyncio.CancelledError):
        await composed.generate(request())

    assert primary_inner.generate_calls == 1
    assert secondary.generate_calls == 0
    assert breaker.state is CircuitState.CLOSED
    assert breaker.consecutive_failures == 0


@pytest.mark.asyncio
async def test_concurrent_request_falls_back_while_half_open_probe_runs() -> None:
    clock = Clock()
    probe_started = asyncio.Event()
    release_probe = asyncio.Event()

    class RecoveringProvider(FakeProvider):
        async def generate(self, request: ChatRequest) -> ChatResponse:
            self.generate_calls += 1
            if self.generate_calls == 1:
                raise ProviderUnavailableError("open circuit")
            probe_started.set()
            await release_probe.wait()
            return response("primary")

    primary_inner = RecoveringProvider()
    primary = CircuitBreakerProvider(
        primary_inner,
        CircuitBreakerPolicy(failure_threshold=1, recovery_timeout=10),
        clock=clock,
    )
    secondary = FakeProvider(response("secondary"), response("secondary"))
    fallback = FallbackProvider([primary, secondary])
    assert (await fallback.generate(request())).provider == "secondary"
    clock.value = 10

    probe = asyncio.create_task(fallback.generate(request()))
    await probe_started.wait()
    concurrent = await fallback.generate(request())
    release_probe.set()
    probe_result = await probe

    assert concurrent.provider == "secondary"
    assert probe_result.provider == "primary"
    assert primary_inner.generate_calls == 2
    assert primary.state is CircuitState.CLOSED


@pytest.mark.asyncio
async def test_routing_decision_and_telemetry_ignore_wrapped_provider_health() -> None:
    clock = Clock()
    inner = FakeProvider(ProviderUnavailableError("down"))
    breaker = CircuitBreakerProvider(
        inner,
        CircuitBreakerPolicy(failure_threshold=1, recovery_timeout=30),
        clock=clock,
    )
    with pytest.raises(ProviderUnavailableError):
        await breaker.generate(request())
    sink = CollectingRoutingSink(fail=True)
    selected = ModelCandidate("selected", breaker, "selected-model", 1, 2, 3)
    other = ModelCandidate("other", FakeProvider(response()), "other-model", 2, 1, 4)
    router = RoutingProvider(CheapPolicy(), [selected, other], telemetry_sink=sink, clock=Clock())

    with pytest.raises(CircuitOpenError):
        await router.generate(request())

    assert len(sink.events) == 1
    event = sink.events[0]
    assert (event.policy, event.candidate, event.model) == ("cheap", "selected", "selected-model")
    assert (event.cost_score, event.latency_score, event.quality_score) == (1, 2, 3)
    assert "Provider" not in repr(event)


@pytest.mark.asyncio
async def test_breaker_stream_delegates_without_changing_circuit_state() -> None:
    inner = FakeProvider(response())
    breaker = CircuitBreakerProvider(inner, CircuitBreakerPolicy(failure_threshold=1))

    chunks = [chunk async for chunk in breaker.stream(request())]

    assert [chunk.content_delta for chunk in chunks] == ["chunk"]
    assert inner.stream_calls == 1
    assert inner.generate_calls == 0
    assert breaker.state is CircuitState.CLOSED
    assert breaker.consecutive_failures == 0
