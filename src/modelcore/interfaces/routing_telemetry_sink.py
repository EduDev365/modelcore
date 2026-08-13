from typing import Protocol

from modelcore.models.routing_telemetry import RoutingTelemetryEvent


class RoutingTelemetrySink(Protocol):
    """Receives safe metadata for successful initial routing decisions."""

    async def emit(self, event: RoutingTelemetryEvent) -> None:
        """Emit one routing event without changing the selected candidate."""
