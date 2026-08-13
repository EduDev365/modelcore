from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RoutingTelemetryEvent:
    """Safe operational metadata for one successful routing decision."""

    policy: str
    candidate: str
    model: str
    candidate_index: int
    candidate_count: int
    duration_ms: float
    cost_score: float
    latency_score: float
    quality_score: float
