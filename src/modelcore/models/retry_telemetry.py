from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetryTelemetryEvent:
    """Safe operational metadata for one provider attempt."""

    provider: str
    model: str
    attempt: int
    max_attempts: int
    outcome: str
    duration_ms: float
    delay_ms: float | None = None
    error_type: str | None = None
