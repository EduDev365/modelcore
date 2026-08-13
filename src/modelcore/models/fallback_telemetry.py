from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FallbackTelemetryEvent:
    """Safe operational metadata for one logical fallback candidate."""

    provider: str
    model: str
    candidate_index: int
    candidate_count: int
    outcome: str
    duration_ms: float
    error_type: str | None = None
