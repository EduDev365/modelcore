import time
from collections.abc import AsyncIterator, Callable, Sequence

from modelcore.exceptions.provider import (
    CircuitOpenError,
    GenerationTimeoutError,
    ModelCoreError,
    ProviderUnavailableError,
    RateLimitError,
)
from modelcore.interfaces.fallback_telemetry_sink import FallbackTelemetrySink
from modelcore.interfaces.llm_provider import LLMProvider
from modelcore.models.chat_request import ChatRequest
from modelcore.models.chat_response import ChatResponse
from modelcore.models.chat_stream_chunk import ChatStreamChunk
from modelcore.models.fallback_telemetry import FallbackTelemetryEvent

FallbackClock = Callable[[], float]

_FALLBACK_ELIGIBLE_ERRORS = (
    CircuitOpenError,
    GenerationTimeoutError,
    ProviderUnavailableError,
    RateLimitError,
)


class NoOpFallbackTelemetrySink:
    """Fallback telemetry sink that intentionally discards events."""

    async def emit(self, event: FallbackTelemetryEvent) -> None:
        return None


class FallbackProvider:
    """Tries chat providers in order after an eligible ModelCore failure."""

    def __init__(
        self,
        providers: Sequence[LLMProvider],
        *,
        provider_names: Sequence[str] | None = None,
        telemetry_sink: FallbackTelemetrySink | None = None,
        clock: FallbackClock = time.monotonic,
    ) -> None:
        normalized_providers = tuple(providers)
        if not normalized_providers:
            raise ValueError("FallbackProvider requires at least one provider")
        if len({id(provider) for provider in normalized_providers}) != len(normalized_providers):
            raise ValueError("FallbackProvider providers must not repeat")
        normalized_names = None if provider_names is None else tuple(provider_names)
        if normalized_names is not None:
            if len(normalized_names) != len(normalized_providers):
                raise ValueError("provider_names must match providers length")
            if any(not name.strip() for name in normalized_names):
                raise ValueError("provider_names cannot contain blank names")
            if len(set(normalized_names)) != len(normalized_names):
                raise ValueError("provider_names must be unique")
        if telemetry_sink is not None and normalized_names is None:
            raise ValueError("provider_names is required when telemetry_sink is configured")
        self._providers = normalized_providers
        self._provider_names = normalized_names
        self._telemetry_sink = telemetry_sink
        self._clock = clock

    async def generate(self, request: ChatRequest) -> ChatResponse:
        for index, provider in enumerate(self._providers):
            started_at = self._clock() if self._telemetry_sink is not None else 0.0
            try:
                response = await provider.generate(request)
            except _FALLBACK_ELIGIBLE_ERRORS as error:
                if index == len(self._providers) - 1:
                    await self._emit_best_effort(
                        self._event(request, index, "exhausted", started_at, type(error).__name__)
                    )
                    raise
                await self._emit_best_effort(self._event(request, index, "fallback", started_at, type(error).__name__))
            except ModelCoreError as error:
                await self._emit_best_effort(self._event(request, index, "error", started_at, type(error).__name__))
                raise
            else:
                await self._emit_best_effort(self._event(request, index, "success", started_at))
                return response

        raise AssertionError("Fallback provider loop completed without returning or raising")

    def stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        """Delegate only to the primary provider; partial streams cannot safely fallback."""
        return self._providers[0].stream(request)

    def _event(
        self,
        request: ChatRequest,
        index: int,
        outcome: str,
        started_at: float,
        error_type: str | None = None,
    ) -> FallbackTelemetryEvent | None:
        if self._telemetry_sink is None:
            return None
        if self._provider_names is None:
            raise AssertionError("provider_names is required for fallback telemetry")
        return FallbackTelemetryEvent(
            provider=self._provider_names[index],
            model=request.model,
            candidate_index=index + 1,
            candidate_count=len(self._providers),
            outcome=outcome,
            duration_ms=(self._clock() - started_at) * 1000,
            error_type=error_type,
        )

    async def _emit_best_effort(self, event: FallbackTelemetryEvent | None) -> None:
        if self._telemetry_sink is None or event is None:
            return
        try:
            await self._telemetry_sink.emit(event)
        except Exception:
            pass
