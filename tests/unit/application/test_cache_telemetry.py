import asyncio
import subprocess
import sys
from collections.abc import Awaitable

import pytest

from modelcore.application import CachingProvider, MemoryCache, ObservableCacheBackend
from modelcore.cache import RedisCache
from modelcore.exceptions import CacheUnavailableError
from modelcore.interfaces import CacheBackend, CacheTelemetrySink
from modelcore.models import CacheTelemetryEvent, ChatRequest, ChatResponse, Message


class SequenceClock:
    def __init__(self, *values: float) -> None:
        self._values = iter(values)

    def __call__(self) -> float:
        return next(self._values)


class CollectingSink:
    def __init__(self) -> None:
        self.events: list[CacheTelemetryEvent] = []

    async def emit(self, event: CacheTelemetryEvent) -> None:
        self.events.append(event)


class FailingSink:
    async def emit(self, event: CacheTelemetryEvent) -> None:
        raise RuntimeError("sink unavailable with redis://user:secret@host/0")


class CancellingSink:
    async def emit(self, event: CacheTelemetryEvent) -> None:
        raise asyncio.CancelledError


class FakeRedisClient:
    def __init__(self) -> None:
        self.values: dict[str, bytes] = {}

    async def get(self, key: str) -> bytes | None:
        return self.values.get(key)

    async def set(self, key: str, value: bytes, *, px: int | None = None) -> object:
        self.values[key] = value
        return True

    async def delete(self, key: str) -> int:
        return int(self.values.pop(key, None) is not None)


class ErrorBackend:
    def __init__(self, error: BaseException) -> None:
        self.error = error

    async def get(self, key: str) -> ChatResponse | None:
        raise self.error

    async def set(self, key: str, value: ChatResponse, ttl: float | None = None) -> None:
        raise self.error


class Provider:
    def __init__(self, response: ChatResponse) -> None:
        self.response = response
        self.calls = 0

    async def generate(self, request: ChatRequest) -> ChatResponse:
        self.calls += 1
        return self.response

    def stream(self, request: ChatRequest) -> Awaitable[None]:
        raise NotImplementedError


def make_response() -> ChatResponse:
    return ChatResponse(
        content="secret generated content",
        model="test-model",
        provider="fake",
        usage=None,
        finish_reason="stop",
    )


@pytest.mark.asyncio
async def test_memory_cache_miss_hit_and_set_emit_safe_events_with_duration() -> None:
    sink = CollectingSink()
    cache: CacheBackend = ObservableCacheBackend(
        MemoryCache(),
        backend_name="memory",
        sink=sink,
        clock=SequenceClock(1.0, 1.01, 2.0, 2.02, 3.0, 3.03),
    )

    assert await cache.get("secret-cache-key") is None
    await cache.set("secret-cache-key", make_response())
    assert await cache.get("secret-cache-key") == make_response()

    assert [(event.operation, event.outcome, event.backend) for event in sink.events] == [
        ("get", "miss", "memory"),
        ("set", "success", "memory"),
        ("get", "hit", "memory"),
    ]
    assert [event.duration_ms for event in sink.events] == pytest.approx([10, 20, 30])
    representation = repr(sink.events)
    assert "secret-cache-key" not in representation
    assert "secret generated content" not in representation


@pytest.mark.asyncio
async def test_redis_cache_hit_and_miss_use_explicit_backend_identity_without_details() -> None:
    sink = CollectingSink()
    client = FakeRedisClient()
    cache = ObservableCacheBackend(
        RedisCache(client), backend_name="redis", sink=sink, clock=SequenceClock(1, 2, 3, 4, 5, 6)
    )

    assert await cache.get("key") is None
    await cache.set("key", make_response())
    assert await cache.get("key") == make_response()

    assert [(event.operation, event.outcome, event.backend) for event in sink.events] == [
        ("get", "miss", "redis"),
        ("set", "success", "redis"),
        ("get", "hit", "redis"),
    ]
    assert "redis://" not in repr(sink.events)


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["get", "set"])
async def test_backend_error_is_observed_and_original_error_propagates(operation: str) -> None:
    error = CacheUnavailableError("redis unavailable at redis://user:secret@host/0")
    sink = CollectingSink()
    cache = ObservableCacheBackend(ErrorBackend(error), backend_name="redis", sink=sink, clock=SequenceClock(1, 1.25))

    with pytest.raises(CacheUnavailableError) as caught:
        if operation == "get":
            await cache.get("secret-key")
        else:
            await cache.set("secret-key", make_response())

    assert caught.value is error
    assert sink.events == [CacheTelemetryEvent(operation, "error", "redis", 250.0, "CacheUnavailableError")]
    assert "secret" not in repr(sink.events)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["hit", "miss"])
async def test_sink_failure_does_not_change_get_result(outcome: str) -> None:
    backend = MemoryCache()
    response = make_response()
    if outcome == "hit":
        await backend.set("key", response)
    cache = ObservableCacheBackend(backend, backend_name="memory", sink=FailingSink(), clock=SequenceClock(1, 2))

    result = await cache.get("key")

    if outcome == "hit":
        assert result is response
    else:
        assert result is None


@pytest.mark.asyncio
async def test_sink_failure_during_backend_error_does_not_mask_original() -> None:
    error = CacheUnavailableError("original")
    cache = ObservableCacheBackend(
        ErrorBackend(error), backend_name="redis", sink=FailingSink(), clock=SequenceClock(1, 2)
    )

    with pytest.raises(CacheUnavailableError) as caught:
        await cache.get("key")
    assert caught.value is error


@pytest.mark.asyncio
async def test_backend_and_sink_cancellation_propagate_without_false_error_event() -> None:
    sink = CollectingSink()
    backend_cancelled = ObservableCacheBackend(
        ErrorBackend(asyncio.CancelledError()), backend_name="redis", sink=sink, clock=SequenceClock(1)
    )
    with pytest.raises(asyncio.CancelledError):
        await backend_cancelled.get("key")
    assert sink.events == []

    sink_cancelled = ObservableCacheBackend(
        MemoryCache(), backend_name="memory", sink=CancellingSink(), clock=SequenceClock(1, 2)
    )
    with pytest.raises(asyncio.CancelledError):
        await sink_cancelled.get("key")


@pytest.mark.asyncio
@pytest.mark.parametrize("backend_name", ["memory", "redis"])
async def test_caching_provider_composes_without_knowing_about_telemetry(backend_name: str) -> None:
    backend: CacheBackend = MemoryCache() if backend_name == "memory" else RedisCache(FakeRedisClient())
    sink = CollectingSink()
    observed = ObservableCacheBackend(backend, backend_name=backend_name, sink=sink)
    provider = Provider(make_response())
    cached = CachingProvider(provider, observed, provider_key="fake")
    request = ChatRequest([Message.user("secret prompt")], model="test-model")

    await cached.generate(request)
    await cached.generate(request)

    assert provider.calls == 1
    assert [event.outcome for event in sink.events] == ["miss", "miss", "success", "hit"]


def test_cache_telemetry_contracts_are_structural_and_backend_name_is_required() -> None:
    sink: CacheTelemetrySink = CollectingSink()
    assert sink is not None
    with pytest.raises(ValueError, match="backend_name cannot be blank"):
        ObservableCacheBackend(MemoryCache(), backend_name=" ")


def test_cache_telemetry_does_not_require_opentelemetry() -> None:
    code = """
import builtins
original_import = builtins.__import__
def blocked_import(name, *args, **kwargs):
    if name == 'opentelemetry' or name.startswith('opentelemetry.'):
        raise AssertionError('OpenTelemetry must remain optional')
    return original_import(name, *args, **kwargs)
builtins.__import__ = blocked_import
from modelcore.application import MemoryCache, ObservableCacheBackend
from modelcore.interfaces import CacheTelemetrySink
from modelcore.models import CacheTelemetryEvent
assert ObservableCacheBackend(MemoryCache(), backend_name='memory')
"""
    subprocess.run([sys.executable, "-c", code], check=True)
