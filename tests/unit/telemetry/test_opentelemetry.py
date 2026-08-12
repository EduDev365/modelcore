import asyncio
import subprocess
import sys

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from modelcore.application.telemetry import TelemetryProvider
from modelcore.models.chat_request import ChatRequest
from modelcore.models.chat_response import ChatResponse
from modelcore.models.message import Message
from modelcore.models.usage import Usage
from modelcore.telemetry.opentelemetry import OpenTelemetrySink


class SequenceClock:
    def __init__(self, *values: float) -> None:
        self._values = iter(values)

    def __call__(self) -> float:
        return next(self._values)


class FakeProvider:
    def __init__(self, outcome: ChatResponse | BaseException) -> None:
        self._outcome = outcome

    async def generate(self, request: ChatRequest) -> ChatResponse:
        if isinstance(self._outcome, BaseException):
            raise self._outcome
        return self._outcome


class FailingTracer:
    def start_span(self, *args: object, **kwargs: object) -> object:
        raise RuntimeError("instrumentation unavailable")


def make_request() -> ChatRequest:
    return ChatRequest(messages=[Message.user("secret prompt")], model="test-model")


def make_response(usage: Usage | None = Usage(3, 2)) -> ChatResponse:
    return ChatResponse(
        content="secret generated content",
        model="test-model",
        provider="openai",
        usage=usage,
        finish_reason="stop",
    )


def make_sink() -> tuple[OpenTelemetrySink, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    return OpenTelemetrySink(tracer_provider.get_tracer("modelcore.tests")), exporter


@pytest.mark.asyncio
async def test_success_event_creates_safe_span_with_usage() -> None:
    sink, exporter = make_sink()
    telemetry = TelemetryProvider(FakeProvider(make_response()), sink, clock=SequenceClock(1, 1.25))

    assert await telemetry.generate(make_request()) == make_response()

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "modelcore.generate"
    assert span.status.status_code is StatusCode.UNSET
    assert dict(span.attributes) == {
        "modelcore.operation": "generate",
        "modelcore.provider": "openai",
        "modelcore.model": "test-model",
        "modelcore.success": True,
        "modelcore.duration_ms": 250.0,
        "modelcore.input_tokens": 3,
        "modelcore.output_tokens": 2,
        "modelcore.total_tokens": 5,
    }
    assert "secret prompt" not in repr(span)
    assert "secret generated content" not in repr(span)


@pytest.mark.asyncio
async def test_success_event_without_usage_omits_usage_attributes() -> None:
    sink, exporter = make_sink()
    telemetry = TelemetryProvider(FakeProvider(make_response(usage=None)), sink, clock=SequenceClock(1, 2))

    await telemetry.generate(make_request())

    attributes = dict(exporter.get_finished_spans()[0].attributes)
    assert "modelcore.input_tokens" not in attributes
    assert "modelcore.output_tokens" not in attributes
    assert "modelcore.total_tokens" not in attributes


@pytest.mark.asyncio
async def test_failure_event_creates_error_span_with_safe_error_type() -> None:
    sink, exporter = make_sink()
    error = RuntimeError("secret api_key=super-secret")
    telemetry = TelemetryProvider(FakeProvider(error), sink, clock=SequenceClock(1, 1.5))

    with pytest.raises(RuntimeError) as raised:
        await telemetry.generate(make_request())

    assert raised.value is error
    span = exporter.get_finished_spans()[0]
    assert span.status.status_code is StatusCode.ERROR
    assert dict(span.attributes) == {
        "modelcore.operation": "generate",
        "modelcore.model": "test-model",
        "modelcore.success": False,
        "modelcore.duration_ms": 500.0,
        "modelcore.error_type": "RuntimeError",
    }
    assert "secret api_key=super-secret" not in repr(span)
    assert not any(key.startswith("exception.") for key in span.attributes)


@pytest.mark.asyncio
async def test_cancelled_error_does_not_create_a_span() -> None:
    sink, exporter = make_sink()
    telemetry = TelemetryProvider(FakeProvider(asyncio.CancelledError()), sink, clock=SequenceClock(1))

    with pytest.raises(asyncio.CancelledError):
        await telemetry.generate(make_request())

    assert exporter.get_finished_spans() == ()


@pytest.mark.asyncio
async def test_instrumentation_failure_is_best_effort() -> None:
    response = make_response()
    telemetry = TelemetryProvider(FakeProvider(response), OpenTelemetrySink(FailingTracer()), clock=SequenceClock(1, 2))  # type: ignore[arg-type]

    assert await telemetry.generate(make_request()) is response


def test_importing_modelcore_does_not_require_opentelemetry() -> None:
    code = """
import builtins
original_import = builtins.__import__
def blocked_import(name, *args, **kwargs):
    if name == 'opentelemetry' or name.startswith('opentelemetry.'):
        raise AssertionError('OpenTelemetry must remain optional')
    return original_import(name, *args, **kwargs)
builtins.__import__ = blocked_import
import modelcore
"""
    subprocess.run([sys.executable, "-c", code], check=True)
