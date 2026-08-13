from modelcore.application import (
    CachingProvider,
    CircuitBreakerPolicy,
    CircuitBreakerProvider,
    CircuitOpenError,
    FallbackProvider,
    ResilientProvider,
    RoutingProvider,
    TelemetryProvider,
)
from modelcore.exceptions import CircuitOpenError as PublicCircuitOpenError
from modelcore.interfaces import CacheBackend, LLMProvider, RoutingTelemetrySink, TelemetrySink
from modelcore.models import ChatRequest, ChatResponse, EmbeddingRequest, Message, RoutingTelemetryEvent


def test_representative_public_imports_are_stable() -> None:
    public_objects = (
        CachingProvider,
        CircuitBreakerPolicy,
        CircuitBreakerProvider,
        FallbackProvider,
        ResilientProvider,
        RoutingProvider,
        TelemetryProvider,
        CacheBackend,
        LLMProvider,
        RoutingTelemetrySink,
        TelemetrySink,
        ChatRequest,
        ChatResponse,
        EmbeddingRequest,
        Message,
        RoutingTelemetryEvent,
    )

    assert all(public_object is not None for public_object in public_objects)
    assert CircuitOpenError is PublicCircuitOpenError
