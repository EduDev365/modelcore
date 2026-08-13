from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CacheTelemetryEvent:
    """Safe operational metadata for one cache backend operation."""

    operation: str
    outcome: str
    backend: str
    duration_ms: float
    error_type: str | None = None
