from typing import Protocol

from modelcore.models.telemetry import GenerationTelemetryEvent


class TelemetrySink(Protocol):
    """Receives safe, provider-agnostic telemetry events."""

    async def emit(self, event: GenerationTelemetryEvent) -> None:
        """Emit one event without changing the observed operation."""
