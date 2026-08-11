import asyncio
import logging
import time
from collections.abc import AsyncIterator, Callable

from modelcore.interfaces.llm_provider import LLMProvider
from modelcore.interfaces.telemetry_sink import TelemetrySink
from modelcore.models.chat_request import ChatRequest
from modelcore.models.chat_response import ChatResponse
from modelcore.models.chat_stream_chunk import ChatStreamChunk
from modelcore.models.telemetry import GenerationTelemetryEvent

Clock = Callable[[], float]


class NoOpTelemetrySink:
    """Default sink that makes telemetry observation optional."""

    async def emit(self, event: GenerationTelemetryEvent) -> None:
        return None


class LoggingTelemetrySink:
    """Standard-library logging sink that emits only operational metadata."""

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    async def emit(self, event: GenerationTelemetryEvent) -> None:
        data = {
            "operation": event.operation,
            "provider": event.provider,
            "model": event.model,
            "duration_ms": event.duration_ms,
            "success": event.success,
            "input_tokens": event.usage.input_tokens if event.usage is not None else None,
            "output_tokens": event.usage.output_tokens if event.usage is not None else None,
            "total_tokens": event.usage.total_tokens if event.usage is not None else None,
            "error_type": event.error_type,
        }
        log = self._logger.info if event.success else self._logger.warning
        log("ModelCore generation telemetry", extra={"modelcore_telemetry": data})


class TelemetryProvider:
    """Observes normal generation through composition without changing outcomes."""

    def __init__(
        self,
        provider: LLMProvider,
        sink: TelemetrySink = NoOpTelemetrySink(),
        *,
        clock: Clock = time.monotonic,
    ) -> None:
        self._provider = provider
        self._sink = sink
        self._clock = clock

    async def generate(self, request: ChatRequest) -> ChatResponse:
        started_at = self._clock()
        try:
            response = await self._provider.generate(request)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            await self._emit_best_effort(
                GenerationTelemetryEvent(
                    operation="generate",
                    provider=None,
                    model=request.model,
                    duration_ms=(self._clock() - started_at) * 1000,
                    success=False,
                    usage=None,
                    error_type=type(error).__name__,
                )
            )
            raise

        await self._emit_best_effort(
            GenerationTelemetryEvent(
                operation="generate",
                provider=response.provider,
                model=response.model,
                duration_ms=(self._clock() - started_at) * 1000,
                success=True,
                usage=response.usage,
                error_type=None,
            )
        )
        return response

    def stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        """Delegate streaming unchanged; chunk telemetry is outside this milestone."""
        return self._provider.stream(request)

    async def _emit_best_effort(self, event: GenerationTelemetryEvent) -> None:
        try:
            await self._sink.emit(event)
        except Exception:
            pass
