import time
from collections.abc import Callable

from modelcore.exceptions import CacheBackendError
from modelcore.interfaces.cache_backend import CacheBackend
from modelcore.interfaces.cache_telemetry_sink import CacheTelemetrySink
from modelcore.models.cache_telemetry import CacheTelemetryEvent
from modelcore.models.chat_response import ChatResponse

CacheClock = Callable[[], float]


class NoOpCacheTelemetrySink:
    """Default sink that makes cache telemetry optional."""

    async def emit(self, event: CacheTelemetryEvent) -> None:
        return None


class ObservableCacheBackend:
    """Observe a cache backend through composition without changing outcomes."""

    def __init__(
        self,
        backend: CacheBackend,
        *,
        backend_name: str,
        sink: CacheTelemetrySink = NoOpCacheTelemetrySink(),
        clock: CacheClock = time.monotonic,
    ) -> None:
        if not backend_name.strip():
            raise ValueError("backend_name cannot be blank")
        self._backend = backend
        self._backend_name = backend_name
        self._sink = sink
        self._clock = clock

    async def get(self, key: str) -> ChatResponse | None:
        started_at = self._clock()
        try:
            response = await self._backend.get(key)
        except CacheBackendError as error:
            await self._emit_best_effort(self._event("get", "error", started_at, type(error).__name__))
            raise

        outcome = "hit" if response is not None else "miss"
        await self._emit_best_effort(self._event("get", outcome, started_at))
        return response

    async def set(self, key: str, value: ChatResponse, ttl: float | None = None) -> None:
        started_at = self._clock()
        try:
            await self._backend.set(key, value, ttl=ttl)
        except CacheBackendError as error:
            await self._emit_best_effort(self._event("set", "error", started_at, type(error).__name__))
            raise

        await self._emit_best_effort(self._event("set", "success", started_at))

    def _event(
        self,
        operation: str,
        outcome: str,
        started_at: float,
        error_type: str | None = None,
    ) -> CacheTelemetryEvent:
        return CacheTelemetryEvent(
            operation=operation,
            outcome=outcome,
            backend=self._backend_name,
            duration_ms=(self._clock() - started_at) * 1000,
            error_type=error_type,
        )

    async def _emit_best_effort(self, event: CacheTelemetryEvent) -> None:
        try:
            await self._sink.emit(event)
        except Exception:
            pass
