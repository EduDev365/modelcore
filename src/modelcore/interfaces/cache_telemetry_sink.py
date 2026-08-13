from typing import Protocol

from modelcore.models.cache_telemetry import CacheTelemetryEvent


class CacheTelemetrySink(Protocol):
    """Receives safe, backend-agnostic cache telemetry events."""

    async def emit(self, event: CacheTelemetryEvent) -> None:
        """Emit one cache event without changing the observed operation."""
