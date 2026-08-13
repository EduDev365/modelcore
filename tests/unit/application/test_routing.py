import asyncio

import pytest

from modelcore.application.routing import (
    BalancedPolicy,
    CheapPolicy,
    FastPolicy,
    ModelCandidate,
    NoOpRoutingTelemetrySink,
    QualityPolicy,
    RoutingProvider,
)
from modelcore.models import RoutingTelemetryEvent
from modelcore.models.chat_request import ChatRequest
from modelcore.models.chat_response import ChatResponse
from modelcore.models.message import Message


class FakeProvider:
    def __init__(self, response: ChatResponse | None = None, error: BaseException | None = None) -> None:
        self.response = response or ChatResponse(
            content="response",
            model="provider-model",
            provider="fake",
            usage=None,
        )
        self.error = error
        self.requests: list[ChatRequest] = []

    async def generate(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return self.response


def make_candidate(
    key: str,
    *,
    provider: FakeProvider | None = None,
    model: str | None = None,
    cost: float = 1,
    latency: float = 1,
    quality: float = 1,
) -> ModelCandidate:
    return ModelCandidate(
        key=key,
        provider=provider or FakeProvider(),
        model=model or f"{key}-model",
        cost_score=cost,
        latency_score=latency,
        quality_score=quality,
    )


def make_request() -> ChatRequest:
    return ChatRequest(messages=[Message.user("Hello")], model="requested-model", temperature=0.2)


class CollectingRoutingSink:
    def __init__(self) -> None:
        self.events: list[RoutingTelemetryEvent] = []

    async def emit(self, event: RoutingTelemetryEvent) -> None:
        self.events.append(event)


class FailingRoutingSink:
    async def emit(self, event: RoutingTelemetryEvent) -> None:
        raise RuntimeError("sink failure")


def test_policies_select_candidates_using_configured_scores() -> None:
    expensive_fast_quality = make_candidate("a", cost=3, latency=1, quality=5)
    cheap_slow = make_candidate("b", cost=1, latency=4, quality=2)
    candidates = [expensive_fast_quality, cheap_slow]

    assert CheapPolicy().select(make_request(), candidates) is cheap_slow
    assert FastPolicy().select(make_request(), candidates) is expensive_fast_quality
    assert QualityPolicy().select(make_request(), candidates) is expensive_fast_quality


def test_builtin_policies_expose_stable_operational_identities() -> None:
    assert [
        CheapPolicy().policy_name,
        FastPolicy().policy_name,
        QualityPolicy().policy_name,
        BalancedPolicy().policy_name,
    ] == [
        "cheap",
        "fast",
        "quality",
        "balanced",
    ]


def test_balanced_policy_uses_quality_minus_cost_minus_latency() -> None:
    first = make_candidate("first", cost=1, latency=3, quality=6)
    second = make_candidate("second", cost=3, latency=1, quality=7)

    assert BalancedPolicy().select(make_request(), [first, second]) is second


@pytest.mark.parametrize("policy", [CheapPolicy(), FastPolicy(), QualityPolicy(), BalancedPolicy()])
def test_policy_ties_preserve_declared_candidate_order(policy: object) -> None:
    first = make_candidate("first", cost=1, latency=1, quality=1)
    second = make_candidate("second", cost=1, latency=1, quality=1)

    assert policy.select(make_request(), [first, second]) is first  # type: ignore[union-attr]


@pytest.mark.parametrize("policy", [CheapPolicy(), FastPolicy(), QualityPolicy(), BalancedPolicy()])
def test_policies_reject_empty_candidate_sequences(policy: object) -> None:
    with pytest.raises(ValueError, match="at least one candidate"):
        policy.select(make_request(), [])  # type: ignore[union-attr]


def test_model_candidate_validates_identity_and_non_negative_finite_scores() -> None:
    with pytest.raises(ValueError, match="key cannot be blank"):
        make_candidate(" ")
    with pytest.raises(ValueError, match="model cannot be blank"):
        make_candidate("candidate", model=" ")
    with pytest.raises(ValueError, match="cost_score must be a non-negative finite number"):
        make_candidate("candidate", cost=-1)
    with pytest.raises(ValueError, match="latency_score must be a non-negative finite number"):
        make_candidate("candidate", latency=float("inf"))


@pytest.mark.asyncio
async def test_router_sends_immutable_request_copy_with_selected_model_and_preserves_response() -> None:
    response = ChatResponse(content="answer", model="selected-model", provider="ollama", usage=None)
    provider = FakeProvider(response=response)
    candidate = make_candidate("selected", provider=provider, model="selected-model", cost=1)
    original_request = make_request()
    router = RoutingProvider(CheapPolicy(), [candidate])

    result = await router.generate(original_request)

    assert result is response
    assert result.provider == "ollama"
    assert provider.requests == [
        ChatRequest(messages=original_request.messages, model="selected-model", temperature=0.2)
    ]
    assert original_request.model == "requested-model"


@pytest.mark.asyncio
async def test_router_emits_selected_candidate_scores_and_duration() -> None:
    sink = CollectingRoutingSink()
    candidate = make_candidate("selected", model="gpt-test", cost=2, latency=3, quality=4)
    clock_values = iter([10.0, 10.125])
    router = RoutingProvider(
        CheapPolicy(),
        [make_candidate("other", cost=3), candidate],
        telemetry_sink=sink,
        clock=lambda: next(clock_values),
    )

    await router.generate(make_request())

    assert sink.events == [RoutingTelemetryEvent("cheap", "selected", "gpt-test", 2, 2, 125.0, 2.0, 3.0, 4.0)]


@pytest.mark.asyncio
async def test_router_sink_failure_does_not_change_generation() -> None:
    provider = FakeProvider()
    candidate = make_candidate("selected", provider=provider)
    router = RoutingProvider(CheapPolicy(), [candidate], telemetry_sink=FailingRoutingSink())

    result = await router.generate(make_request())

    assert result is provider.response
    assert provider.requests[0].model == candidate.model


def test_external_policy_requires_explicit_name_only_with_telemetry() -> None:
    class ExternalPolicy:
        def select(self, request: ChatRequest, candidates: list[ModelCandidate]) -> ModelCandidate:
            return candidates[0]

    candidate = make_candidate("candidate")
    with pytest.raises(ValueError, match="policy_name is required"):
        RoutingProvider(ExternalPolicy(), [candidate], telemetry_sink=NoOpRoutingTelemetrySink())

    router = RoutingProvider(
        ExternalPolicy(), [candidate], policy_name=" external-policy ", telemetry_sink=CollectingRoutingSink()
    )
    assert router is not None


def test_policy_name_rejects_blank_values() -> None:
    with pytest.raises(ValueError, match="policy_name cannot be blank"):
        RoutingProvider(CheapPolicy(), [make_candidate("candidate")], policy_name=" ")


@pytest.mark.asyncio
async def test_router_propagates_chosen_provider_errors_and_cancellation() -> None:
    request = make_request()
    error_provider = FakeProvider(error=RuntimeError("bug"))
    router = RoutingProvider(CheapPolicy(), [make_candidate("failing", provider=error_provider)])

    with pytest.raises(RuntimeError, match="bug"):
        await router.generate(request)

    cancelled_provider = FakeProvider(error=asyncio.CancelledError())
    cancelled_router = RoutingProvider(CheapPolicy(), [make_candidate("cancelled", provider=cancelled_provider)])
    with pytest.raises(asyncio.CancelledError):
        await cancelled_router.generate(request)


def test_router_rejects_empty_or_duplicate_candidate_configuration() -> None:
    candidate = make_candidate("candidate")

    with pytest.raises(ValueError, match="at least one candidate"):
        RoutingProvider(CheapPolicy(), [])
    with pytest.raises(ValueError, match="keys must be unique"):
        RoutingProvider(CheapPolicy(), [candidate, candidate])


@pytest.mark.asyncio
async def test_router_rejects_policy_selection_outside_configured_candidates() -> None:
    class ForeignPolicy:
        def select(self, request: ChatRequest, candidates: list[ModelCandidate]) -> ModelCandidate:
            return make_candidate("foreign")

    router = RoutingProvider(ForeignPolicy(), [make_candidate("configured")])

    with pytest.raises(ValueError, match="configured candidate"):
        await router.generate(make_request())
