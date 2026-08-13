import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, StatusCode

from modelcore.models import CacheTelemetryEvent
from modelcore.telemetry.opentelemetry import OpenTelemetryCacheSink


def make_sink() -> tuple[OpenTelemetryCacheSink, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    return OpenTelemetryCacheSink(tracer_provider.get_tracer("modelcore.cache.tests")), exporter


@pytest.mark.asyncio
async def test_cache_event_creates_safe_internal_span() -> None:
    sink, exporter = make_sink()

    await sink.emit(CacheTelemetryEvent("get", "hit", "redis", 12.5))

    span = exporter.get_finished_spans()[0]
    assert span.name == "modelcore.cache.get"
    assert span.kind is SpanKind.INTERNAL
    assert span.status.status_code is StatusCode.UNSET
    assert dict(span.attributes) == {
        "modelcore.cache.operation": "get",
        "modelcore.cache.outcome": "hit",
        "modelcore.cache.backend": "redis",
        "modelcore.cache.duration_ms": 12.5,
    }
    representation = repr(span)
    assert "cache-key" not in representation
    assert "generated content" not in representation
    assert "redis://" not in representation


@pytest.mark.asyncio
async def test_cache_error_event_creates_error_span_without_raw_exception() -> None:
    sink, exporter = make_sink()

    await sink.emit(CacheTelemetryEvent("set", "error", "redis", 5.0, "CacheUnavailableError"))

    span = exporter.get_finished_spans()[0]
    assert span.name == "modelcore.cache.set"
    assert span.status.status_code is StatusCode.ERROR
    assert dict(span.attributes) == {
        "modelcore.cache.operation": "set",
        "modelcore.cache.outcome": "error",
        "modelcore.cache.backend": "redis",
        "modelcore.cache.duration_ms": 5.0,
        "modelcore.cache.error_type": "CacheUnavailableError",
    }
    assert not any(name.startswith("exception.") for name in span.attributes)
