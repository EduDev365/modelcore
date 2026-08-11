import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Protocol

from modelcore.interfaces.llm_provider import LLMProvider
from modelcore.models.chat_request import ChatRequest
from modelcore.models.chat_response import ChatResponse


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

    def select(self, request: ChatRequest, candidates: Sequence[ModelCandidate]) -> ModelCandidate:
        return min(_require_candidates(candidates), key=lambda candidate: candidate.cost_score)


class FastPolicy:
    """Select the candidate with the lowest configured latency score."""

    def select(self, request: ChatRequest, candidates: Sequence[ModelCandidate]) -> ModelCandidate:
        return min(_require_candidates(candidates), key=lambda candidate: candidate.latency_score)


class QualityPolicy:
    """Select the candidate with the highest configured quality score."""

    def select(self, request: ChatRequest, candidates: Sequence[ModelCandidate]) -> ModelCandidate:
        return max(_require_candidates(candidates), key=lambda candidate: candidate.quality_score)


class BalancedPolicy:
    """Select the highest quality-minus-cost-minus-latency candidate."""

    def select(self, request: ChatRequest, candidates: Sequence[ModelCandidate]) -> ModelCandidate:
        return max(
            _require_candidates(candidates),
            key=lambda candidate: candidate.quality_score - candidate.cost_score - candidate.latency_score,
        )


class RoutingProvider:
    """Routes normal generation to a configured provider/model candidate."""

    def __init__(self, policy: RoutingPolicy, candidates: Sequence[ModelCandidate]) -> None:
        normalized_candidates = tuple(_require_candidates(candidates))
        if len({candidate.key for candidate in normalized_candidates}) != len(normalized_candidates):
            raise ValueError("candidate keys must be unique")
        self._policy = policy
        self._candidates = normalized_candidates

    async def generate(self, request: ChatRequest) -> ChatResponse:
        candidate = self._policy.select(request, self._candidates)
        if not any(candidate is configured for configured in self._candidates):
            raise ValueError("Routing policy selected a candidate outside the configured candidates")
        routed_request = replace(request, model=candidate.model)
        return await candidate.provider.generate(routed_request)


def _require_candidates(candidates: Sequence[ModelCandidate]) -> Sequence[ModelCandidate]:
    if not candidates:
        raise ValueError("Routing requires at least one candidate")
    return candidates
