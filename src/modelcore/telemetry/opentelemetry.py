"""OpenTelemetry adapter for ModelCore's internal telemetry events.

Install with ``modelcore[otel]``. Applications own tracer-provider and exporter
configuration; this module never modifies global OpenTelemetry state.
"""

from opentelemetry.trace import SpanKind, Status, StatusCode, Tracer

from modelcore.models.telemetry import GenerationTelemetryEvent


class OpenTelemetrySink:
    """Convert safe final generation events into OpenTelemetry spans."""

    def __init__(self, tracer: Tracer) -> None:
        self._tracer = tracer

    async def emit(self, event: GenerationTelemetryEvent) -> None:
        span = self._tracer.start_span(f"modelcore.{event.operation}", kind=SpanKind.CLIENT)
        try:
            for name, value in _attributes(event).items():
                span.set_attribute(name, value)
            if not event.success:
                span.set_status(Status(StatusCode.ERROR))
        finally:
            span.end()


def _attributes(event: GenerationTelemetryEvent) -> dict[str, str | float | int | bool]:
    attributes: dict[str, str | float | int | bool] = {
        "modelcore.operation": event.operation,
        "modelcore.success": event.success,
        "modelcore.duration_ms": event.duration_ms,
    }
    if event.provider is not None:
        attributes["modelcore.provider"] = event.provider
    if event.model is not None:
        attributes["modelcore.model"] = event.model
    if event.usage is not None:
        attributes["modelcore.input_tokens"] = event.usage.input_tokens
        attributes["modelcore.output_tokens"] = event.usage.output_tokens
        if event.usage.total_tokens is not None:
            attributes["modelcore.total_tokens"] = event.usage.total_tokens
    if event.error_type is not None:
        attributes["modelcore.error_type"] = event.error_type
    return attributes
