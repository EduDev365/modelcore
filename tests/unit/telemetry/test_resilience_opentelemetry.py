import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, StatusCode

from modelcore.models import FallbackTelemetryEvent, RetryTelemetryEvent
from modelcore.telemetry.opentelemetry import OpenTelemetryFallbackSink, OpenTelemetryRetrySink


def make_tracer() -> tuple[TracerProvider, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    return tracer_provider, exporter


@pytest.mark.asyncio
async def test_retry_event_creates_safe_internal_span() -> None:
    tracer_provider, exporter = make_tracer()
    sink = OpenTelemetryRetrySink(tracer_provider.get_tracer("modelcore.retry.tests"))

    await sink.emit(RetryTelemetryEvent("openai", "gpt-test", 1, 3, "retry", 12.5, 500.0, "RateLimitError"))

    span = exporter.get_finished_spans()[0]
    assert span.name == "modelcore.retry"
    assert span.kind is SpanKind.INTERNAL
    assert span.status.status_code is StatusCode.ERROR
    assert dict(span.attributes) == {
        "modelcore.retry.provider": "openai",
        "modelcore.retry.model": "gpt-test",
        "modelcore.retry.attempt": 1,
        "modelcore.retry.max_attempts": 3,
        "modelcore.retry.outcome": "retry",
        "modelcore.retry.duration_ms": 12.5,
        "modelcore.retry.delay_ms": 500.0,
        "modelcore.retry.error_type": "RateLimitError",
    }
    representation = repr(span)
    assert "generated content" not in representation
    assert "api-key" not in representation
    assert "https://provider.example" not in representation
    assert not any(name.startswith("exception.") for name in span.attributes)


@pytest.mark.asyncio
async def test_retry_success_span_has_no_error_or_delay() -> None:
    tracer_provider, exporter = make_tracer()
    sink = OpenTelemetryRetrySink(tracer_provider.get_tracer("modelcore.retry.tests"))

    await sink.emit(RetryTelemetryEvent("ollama", "local-model", 2, 3, "success", 4.0))

    span = exporter.get_finished_spans()[0]
    assert span.status.status_code is StatusCode.UNSET
    assert "modelcore.retry.error_type" not in span.attributes
    assert "modelcore.retry.delay_ms" not in span.attributes


@pytest.mark.asyncio
async def test_fallback_event_creates_safe_internal_span() -> None:
    tracer_provider, exporter = make_tracer()
    sink = OpenTelemetryFallbackSink(tracer_provider.get_tracer("modelcore.fallback.tests"))

    await sink.emit(FallbackTelemetryEvent("openai", "gpt-test", 1, 2, "fallback", 8.0, "ProviderUnavailableError"))

    span = exporter.get_finished_spans()[0]
    assert span.name == "modelcore.fallback"
    assert span.kind is SpanKind.INTERNAL
    assert span.status.status_code is StatusCode.ERROR
    assert dict(span.attributes) == {
        "modelcore.fallback.provider": "openai",
        "modelcore.fallback.model": "gpt-test",
        "modelcore.fallback.candidate_index": 1,
        "modelcore.fallback.candidate_count": 2,
        "modelcore.fallback.outcome": "fallback",
        "modelcore.fallback.duration_ms": 8.0,
        "modelcore.fallback.error_type": "ProviderUnavailableError",
    }
    assert not any(name.startswith("exception.") for name in span.attributes)


@pytest.mark.asyncio
async def test_fallback_success_uses_configured_candidate_identity() -> None:
    tracer_provider, exporter = make_tracer()
    sink = OpenTelemetryFallbackSink(tracer_provider.get_tracer("modelcore.fallback.tests"))

    await sink.emit(FallbackTelemetryEvent("ollama", "local-model", 2, 2, "success", 3.0))

    span = exporter.get_finished_spans()[0]
    assert span.status.status_code is StatusCode.UNSET
    assert span.attributes["modelcore.fallback.provider"] == "ollama"
    assert "modelcore.fallback.error_type" not in span.attributes
