from typing import Protocol

from modelcore.models.fallback_telemetry import FallbackTelemetryEvent


class FallbackTelemetrySink(Protocol):
    """Receives safe fallback candidate telemetry events."""

    async def emit(self, event: FallbackTelemetryEvent) -> None:
        """Emit one fallback event without changing fallback policy."""
