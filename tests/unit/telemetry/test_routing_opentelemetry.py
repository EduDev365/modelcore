import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from modelcore.models import RoutingTelemetryEvent
from modelcore.telemetry.opentelemetry import OpenTelemetryRoutingSink


@pytest.mark.asyncio
async def test_routing_opentelemetry_sink_emits_internal_safe_span() -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    sink = OpenTelemetryRoutingSink(provider.get_tracer("modelcore.routing.tests"))

    await sink.emit(RoutingTelemetryEvent("cheap", "openai-cheap", "gpt-test", 1, 2, 4.5, 1.0, 2.0, 3.0))

    span = exporter.get_finished_spans()[0]
    assert span.name == "modelcore.routing"
    assert span.kind.name == "INTERNAL"
    assert span.attributes["modelcore.routing.policy"] == "cheap"
    assert span.attributes["modelcore.routing.candidate"] == "openai-cheap"
    assert span.attributes["modelcore.routing.model"] == "gpt-test"
    assert span.attributes["modelcore.routing.candidate_index"] == 1
    assert span.attributes["modelcore.routing.candidate_count"] == 2
    assert "prompt" not in str(span.attributes)
