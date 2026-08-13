"""OpenTelemetry adapter for ModelCore's internal telemetry events.

Install with ``modelcore[otel]``. Applications own tracer-provider and exporter
configuration; this module never modifies global OpenTelemetry state.
"""

from opentelemetry.trace import SpanKind, Status, StatusCode, Tracer

from modelcore.models.cache_telemetry import CacheTelemetryEvent
from modelcore.models.fallback_telemetry import FallbackTelemetryEvent
from modelcore.models.retry_telemetry import RetryTelemetryEvent
from modelcore.models.routing_telemetry import RoutingTelemetryEvent
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


class OpenTelemetryCacheSink:
    """Convert safe cache operation events into OpenTelemetry spans."""

    def __init__(self, tracer: Tracer) -> None:
        self._tracer = tracer

    async def emit(self, event: CacheTelemetryEvent) -> None:
        span = self._tracer.start_span(f"modelcore.cache.{event.operation}", kind=SpanKind.INTERNAL)
        try:
            span.set_attribute("modelcore.cache.operation", event.operation)
            span.set_attribute("modelcore.cache.outcome", event.outcome)
            span.set_attribute("modelcore.cache.backend", event.backend)
            span.set_attribute("modelcore.cache.duration_ms", event.duration_ms)
            if event.error_type is not None:
                span.set_attribute("modelcore.cache.error_type", event.error_type)
                span.set_status(Status(StatusCode.ERROR))
        finally:
            span.end()


class OpenTelemetryRetrySink:
    """Convert safe retry attempt events into internal OpenTelemetry spans."""

    def __init__(self, tracer: Tracer) -> None:
        self._tracer = tracer

    async def emit(self, event: RetryTelemetryEvent) -> None:
        span = self._tracer.start_span("modelcore.retry", kind=SpanKind.INTERNAL)
        try:
            span.set_attribute("modelcore.retry.provider", event.provider)
            span.set_attribute("modelcore.retry.model", event.model)
            span.set_attribute("modelcore.retry.attempt", event.attempt)
            span.set_attribute("modelcore.retry.max_attempts", event.max_attempts)
            span.set_attribute("modelcore.retry.outcome", event.outcome)
            span.set_attribute("modelcore.retry.duration_ms", event.duration_ms)
            if event.delay_ms is not None:
                span.set_attribute("modelcore.retry.delay_ms", event.delay_ms)
            if event.error_type is not None:
                span.set_attribute("modelcore.retry.error_type", event.error_type)
                span.set_status(Status(StatusCode.ERROR))
        finally:
            span.end()


class OpenTelemetryFallbackSink:
    """Convert safe fallback candidate events into internal OpenTelemetry spans."""

    def __init__(self, tracer: Tracer) -> None:
        self._tracer = tracer

    async def emit(self, event: FallbackTelemetryEvent) -> None:
        span = self._tracer.start_span("modelcore.fallback", kind=SpanKind.INTERNAL)
        try:
            span.set_attribute("modelcore.fallback.provider", event.provider)
            span.set_attribute("modelcore.fallback.model", event.model)
            span.set_attribute("modelcore.fallback.candidate_index", event.candidate_index)
            span.set_attribute("modelcore.fallback.candidate_count", event.candidate_count)
            span.set_attribute("modelcore.fallback.outcome", event.outcome)
            span.set_attribute("modelcore.fallback.duration_ms", event.duration_ms)
            if event.error_type is not None:
                span.set_attribute("modelcore.fallback.error_type", event.error_type)
                span.set_status(Status(StatusCode.ERROR))
        finally:
            span.end()


class OpenTelemetryRoutingSink:
    """Convert initial routing decisions into internal OpenTelemetry spans."""

    def __init__(self, tracer: Tracer) -> None:
        self._tracer = tracer

    async def emit(self, event: RoutingTelemetryEvent) -> None:
        span = self._tracer.start_span("modelcore.routing", kind=SpanKind.INTERNAL)
        try:
            attributes: dict[str, str | float | int] = {
                "modelcore.routing.policy": event.policy,
                "modelcore.routing.candidate": event.candidate,
                "modelcore.routing.model": event.model,
                "modelcore.routing.candidate_index": event.candidate_index,
                "modelcore.routing.candidate_count": event.candidate_count,
                "modelcore.routing.duration_ms": event.duration_ms,
                "modelcore.routing.cost_score": event.cost_score,
                "modelcore.routing.latency_score": event.latency_score,
                "modelcore.routing.quality_score": event.quality_score,
            }
            for name, value in attributes.items():
                span.set_attribute(name, value)
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
