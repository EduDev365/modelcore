from dataclasses import dataclass

from modelcore.models.usage import Usage


@dataclass(frozen=True, slots=True)
class GenerationTelemetryEvent:
    """Safe operational metadata emitted for one normal generation attempt."""

    operation: str
    provider: str | None
    model: str | None
    duration_ms: float
    success: bool
    usage: Usage | None
    error_type: str | None
