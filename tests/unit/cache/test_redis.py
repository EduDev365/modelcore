import asyncio
import inspect
import json
import subprocess
import sys
from collections.abc import AsyncIterator

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import ResponseError
from redis.exceptions import TimeoutError as RedisTimeoutError

from modelcore.application import CachingProvider, ResilientProvider, RetryPolicy
from modelcore.cache import RedisCache
from modelcore.exceptions import CacheBackendError, CacheUnavailableError
from modelcore.interfaces import CacheBackend
from modelcore.models import ChatRequest, ChatResponse, ChatStreamChunk, Message, Usage


class FakeRedisClient:
    def __init__(self) -> None:
        self.values: dict[str, bytes | str] = {}
        self.set_calls: list[tuple[str, bytes, int | None]] = []
        self.delete_calls: list[str] = []
        self.get_error: BaseException | None = None
        self.set_error: BaseException | None = None
        self.delete_error: BaseException | None = None

    async def get(self, key: str) -> bytes | str | None:
        if self.get_error is not None:
            raise self.get_error
        return self.values.get(key)

    async def set(self, key: str, value: bytes, *, px: int | None = None) -> object:
        if self.set_error is not None:
            raise self.set_error
        self.set_calls.append((key, value, px))
        self.values[key] = value
        return True

    async def delete(self, key: str) -> int:
        if self.delete_error is not None:
            raise self.delete_error
        self.delete_calls.append(key)
        return int(self.values.pop(key, None) is not None)


def make_response(*, usage: Usage | None = None, finish_reason: str | None = "stop") -> ChatResponse:
    return ChatResponse(
        content="Olá, mundo",
        model="gpt-test",
        provider="fake-provider",
        usage=usage,
        finish_reason=finish_reason,
    )


@pytest.mark.asyncio
async def test_redis_cache_is_a_structural_cache_backend_and_returns_miss() -> None:
    backend: CacheBackend = RedisCache(FakeRedisClient())

    assert await backend.get("missing") is None


@pytest.mark.asyncio
async def test_redis_cache_serializes_versioned_json_and_uses_namespace() -> None:
    client = FakeRedisClient()
    cache = RedisCache(client, namespace="application:modelcore:")
    response = make_response(usage=Usage(input_tokens=3, output_tokens=2))

    await cache.set("digest", response)

    key, encoded, ttl = client.set_calls[0]
    payload = json.loads(encoded)
    assert key == "application:modelcore:digest"
    assert ttl is None
    assert payload == {
        "version": 1,
        "response": {
            "content": "Olá, mundo",
            "model": "gpt-test",
            "provider": "fake-provider",
            "finish_reason": "stop",
            "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
        },
    }
    assert "request" not in payload


@pytest.mark.asyncio
@pytest.mark.parametrize("as_text", [False, True])
async def test_redis_cache_deserializes_bytes_or_text_with_usage(as_text: bool) -> None:
    client = FakeRedisClient()
    cache = RedisCache(client)
    original = make_response(usage=Usage(input_tokens=7, output_tokens=4), finish_reason="length")
    await cache.set("key", original)
    if as_text:
        stored = client.values["modelcore:key"]
        assert isinstance(stored, bytes)
        client.values["modelcore:key"] = stored.decode("utf-8")

    restored = await cache.get("key")

    assert restored == original
    assert restored is not original
    assert restored is not None
    assert restored.usage == Usage(input_tokens=7, output_tokens=4)
    assert restored.finish_reason == "length"
    assert restored.provider == "fake-provider"
    assert restored.model == "gpt-test"


@pytest.mark.asyncio
async def test_redis_cache_preserves_absent_usage_and_finish_reason() -> None:
    client = FakeRedisClient()
    cache = RedisCache(client)
    original = make_response(usage=None, finish_reason=None)

    await cache.set("key", original)

    assert await cache.get("key") == original


@pytest.mark.asyncio
async def test_redis_cache_uses_native_millisecond_ttl() -> None:
    client = FakeRedisClient()
    cache = RedisCache(client)

    await cache.set("fractional", make_response(), ttl=1.25)
    await cache.set("tiny", make_response(), ttl=0.0001)

    assert client.set_calls[0][2] == 1250
    assert client.set_calls[1][2] == 1


@pytest.mark.asyncio
async def test_redis_cache_zero_ttl_removes_existing_value_and_negative_ttl_is_rejected() -> None:
    client = FakeRedisClient()
    cache = RedisCache(client)
    await cache.set("key", make_response())

    await cache.set("key", make_response(), ttl=0)

    assert client.delete_calls == ["modelcore:key"]
    assert await cache.get("key") is None
    with pytest.raises(ValueError, match="ttl cannot be negative"):
        await cache.set("key", make_response(), ttl=-0.1)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        b"not-json",
        b'{"version":2,"response":{}}',
        b'{"version":1,"response":{"content":42}}',
        b'{"version":1,"response":{"content":"x","model":"m","provider":"p","usage":{"input_tokens":true,"output_tokens":1}}}',
    ],
)
async def test_redis_cache_treats_invalid_or_unknown_payload_as_miss(payload: bytes) -> None:
    client = FakeRedisClient()
    client.values["modelcore:key"] = payload

    assert await RedisCache(client).get("key") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [RedisConnectionError("offline"), RedisTimeoutError("timeout")])
async def test_redis_cache_normalizes_get_unavailability(error: Exception) -> None:
    client = FakeRedisClient()
    client.get_error = error

    with pytest.raises(CacheUnavailableError, match="Redis cache is unavailable") as caught:
        await RedisCache(client).get("key")
    assert caught.value.__cause__ is error


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["set", "delete"])
async def test_redis_cache_normalizes_set_unavailability(operation: str) -> None:
    client = FakeRedisClient()
    error = RedisConnectionError("offline")
    if operation == "set":
        client.set_error = error
        ttl = None
    else:
        client.delete_error = error
        ttl = 0

    with pytest.raises(CacheUnavailableError) as caught:
        await RedisCache(client).set("key", make_response(), ttl=ttl)
    assert caught.value.__cause__ is error


@pytest.mark.asyncio
async def test_redis_cache_normalizes_other_redis_errors_without_hiding_programming_errors() -> None:
    client = FakeRedisClient()
    redis_error = ResponseError("invalid Redis command response")
    client.set_error = redis_error

    with pytest.raises(CacheBackendError, match="Redis cache set failed") as caught:
        await RedisCache(client).set("key", make_response())
    assert caught.value.__cause__ is redis_error

    programming_error = RuntimeError("client bug")
    client.get_error = programming_error
    with pytest.raises(RuntimeError, match="client bug"):
        await RedisCache(client).get("key")


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["get", "set", "delete"])
async def test_redis_cache_propagates_cancellation(operation: str) -> None:
    client = FakeRedisClient()
    setattr(client, f"{operation}_error", asyncio.CancelledError())
    cache = RedisCache(client)

    with pytest.raises(asyncio.CancelledError):
        if operation == "get":
            await cache.get("key")
        else:
            await cache.set("key", make_response(), ttl=0 if operation == "delete" else None)


@pytest.mark.asyncio
async def test_redis_cache_composes_with_caching_and_resilience() -> None:
    class Provider:
        def __init__(self) -> None:
            self.calls = 0

        async def generate(self, request: ChatRequest) -> ChatResponse:
            self.calls += 1
            return make_response()

        def stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
            async def chunks() -> AsyncIterator[ChatStreamChunk]:
                yield ChatStreamChunk(content_delta="x", model=request.model, provider="fake-provider")

            return chunks()

    provider = Provider()
    resilient = ResilientProvider(provider, RetryPolicy(max_attempts=1))
    cached = CachingProvider(resilient, RedisCache(FakeRedisClient()), provider_key="fake")
    request = ChatRequest([Message.user("hello")], model="gpt-test")

    assert await cached.generate(request) == await cached.generate(request)
    assert provider.calls == 1


def test_redis_cache_source_uses_json_not_unsafe_object_serialization() -> None:
    source = inspect.getsource(sys.modules[RedisCache.__module__])

    assert "pickle" not in source
    assert "eval(" not in source


def test_modelcore_import_does_not_require_optional_redis_package() -> None:
    script = """
import builtins
real_import = builtins.__import__
def blocked(name, *args, **kwargs):
    if name == 'redis' or name.startswith('redis.'):
        raise ModuleNotFoundError('redis intentionally unavailable')
    return real_import(name, *args, **kwargs)
builtins.__import__ = blocked
import modelcore
from modelcore.cache import RedisCache
assert RedisCache is not None
"""
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)

    assert completed.returncode == 0, completed.stderr
