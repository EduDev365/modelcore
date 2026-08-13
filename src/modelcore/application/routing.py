import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Protocol

from modelcore.interfaces.llm_provider import LLMProvider
from modelcore.interfaces.routing_telemetry_sink import RoutingTelemetrySink
from modelcore.models.chat_request import ChatRequest
from modelcore.models.chat_response import ChatResponse
from modelcore.models.routing_telemetry import RoutingTelemetryEvent

RoutingClock = Callable[[], float]


class NoOpRoutingTelemetrySink:
    """Default routing sink that intentionally discards events."""

    async def emit(self, event: RoutingTelemetryEvent) -> None:
        return None


@dataclass(frozen=True, slots=True)
class ModelCandidate:
    """Configured provider, model, and relative routing metadata."""

    key: str
    provider: LLMProvider
    model: str
    cost_score: float
    latency_score: float
    quality_score: float

    def __post_init__(self) -> None:
        if not self.key.strip():
            raise ValueError("key cannot be blank")
        if not self.model.strip():
            raise ValueError("model cannot be blank")
        for field_name in ("cost_score", "latency_score", "quality_score"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative finite number")
            object.__setattr__(self, field_name, float(value))


class RoutingPolicy(Protocol):
    """Selects one configured candidate without performing generation."""

    def select(self, request: ChatRequest, candidates: Sequence[ModelCandidate]) -> ModelCandidate:
        """Return a candidate from the declared sequence."""


class CheapPolicy:
    """Select the candidate with the lowest configured cost score."""

    @property
    def policy_name(self) -> str:
        return "cheap"

    def select(self, request: ChatRequest, candidates: Sequence[ModelCandidate]) -> ModelCandidate:
        return min(_require_candidates(candidates), key=lambda candidate: candidate.cost_score)


class FastPolicy:
    """Select the candidate with the lowest configured latency score."""

    @property
    def policy_name(self) -> str:
        return "fast"

    def select(self, request: ChatRequest, candidates: Sequence[ModelCandidate]) -> ModelCandidate:
        return min(_require_candidates(candidates), key=lambda candidate: candidate.latency_score)


class QualityPolicy:
    """Select the candidate with the highest configured quality score."""

    @property
    def policy_name(self) -> str:
        return "quality"

    def select(self, request: ChatRequest, candidates: Sequence[ModelCandidate]) -> ModelCandidate:
        return max(_require_candidates(candidates), key=lambda candidate: candidate.quality_score)


class BalancedPolicy:
    """Select the highest quality-minus-cost-minus-latency candidate."""

    @property
    def policy_name(self) -> str:
        return "balanced"

    def select(self, request: ChatRequest, candidates: Sequence[ModelCandidate]) -> ModelCandidate:
        return max(
            _require_candidates(candidates),
            key=lambda candidate: candidate.quality_score - candidate.cost_score - candidate.latency_score,
        )


class RoutingProvider:
    """Routes normal generation to a configured provider/model candidate."""

    def __init__(
        self,
        policy: RoutingPolicy,
        candidates: Sequence[ModelCandidate],
        *,
        policy_name: str | None = None,
        telemetry_sink: RoutingTelemetrySink | None = None,
        clock: RoutingClock = time.monotonic,
    ) -> None:
        normalized_candidates = tuple(_require_candidates(candidates))
        if len({candidate.key for candidate in normalized_candidates}) != len(normalized_candidates):
            raise ValueError("candidate keys must be unique")
        self._policy = policy
        self._candidates = normalized_candidates
        self._policy_name = _resolve_policy_name(policy, policy_name, telemetry_sink is not None)
        self._telemetry_sink = telemetry_sink
        self._clock = clock

    async def generate(self, request: ChatRequest) -> ChatResponse:
        started_at = self._clock() if self._telemetry_sink is not None else 0.0
        candidate = self._policy.select(request, self._candidates)
        try:
            candidate_index = next(
                index for index, configured in enumerate(self._candidates) if candidate is configured
            )
        except StopIteration:
            raise ValueError("Routing policy selected a candidate outside the configured candidates")
        if self._telemetry_sink is not None:
            assert self._policy_name is not None
            await self._emit_best_effort(
                RoutingTelemetryEvent(
                    policy=self._policy_name,
                    candidate=candidate.key,
                    model=candidate.model,
                    candidate_index=candidate_index + 1,
                    candidate_count=len(self._candidates),
                    duration_ms=(self._clock() - started_at) * 1000,
                    cost_score=candidate.cost_score,
                    latency_score=candidate.latency_score,
                    quality_score=candidate.quality_score,
                )
            )
        routed_request = replace(request, model=candidate.model)
        return await candidate.provider.generate(routed_request)

    async def _emit_best_effort(self, event: RoutingTelemetryEvent) -> None:
        if self._telemetry_sink is None:
            return
        try:
            await self._telemetry_sink.emit(event)
        except Exception:
            pass


def _resolve_policy_name(policy: RoutingPolicy, policy_name: str | None, telemetry_enabled: bool) -> str | None:
    if policy_name is not None:
        if not policy_name.strip():
            raise ValueError("policy_name cannot be blank")
        return policy_name.strip()
    if type(policy) is CheapPolicy:
        return policy.policy_name
    if type(policy) is FastPolicy:
        return policy.policy_name
    if type(policy) is QualityPolicy:
        return policy.policy_name
    if type(policy) is BalancedPolicy:
        return policy.policy_name
    if telemetry_enabled:
        raise ValueError("policy_name is required when telemetry_sink is configured for an external policy")
    return None


def _require_candidates(candidates: Sequence[ModelCandidate]) -> Sequence[ModelCandidate]:
    if not candidates:
        raise ValueError("Routing requires at least one candidate")
    return candidates
