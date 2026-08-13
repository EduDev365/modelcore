from typing import Protocol

from modelcore.models.retry_telemetry import RetryTelemetryEvent


class RetryTelemetrySink(Protocol):
    """Receives safe retry attempt telemetry events."""

    async def emit(self, event: RetryTelemetryEvent) -> None:
        """Emit one retry event without changing retry policy."""
