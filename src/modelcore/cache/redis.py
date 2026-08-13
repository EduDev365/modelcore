import json
import math
from collections.abc import Awaitable
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from redis.exceptions import ConnectionError as RedisConnectionError
    from redis.exceptions import RedisError
    from redis.exceptions import TimeoutError as RedisTimeoutError
else:
    try:
        from redis.exceptions import ConnectionError as RedisConnectionError
        from redis.exceptions import RedisError
        from redis.exceptions import TimeoutError as RedisTimeoutError
    except ModuleNotFoundError:  # pragma: no cover - exercised in an isolated subprocess

        class RedisError(Exception):
            """Fallback used when the optional redis package is absent."""

        class RedisConnectionError(RedisError):
            pass

        class RedisTimeoutError(RedisError):
            pass


from modelcore.exceptions import CacheBackendError, CacheUnavailableError
from modelcore.models import ChatResponse, Usage

_PAYLOAD_VERSION = 1


class AsyncRedisClient(Protocol):
    """Commands required from an asynchronous Redis client."""

    def get(self, key: str) -> Awaitable[bytes | str | None]: ...

    def set(self, key: str, value: bytes, *, px: int | None = None) -> Awaitable[object]: ...

    def delete(self, key: str) -> Awaitable[int]: ...


class RedisCache:
    """Redis-backed storage for normalized chat responses.

    Values are persisted as versioned JSON. Infrastructure failures are
    normalized and propagated; callers choose any best-effort policy.
    """

    def __init__(self, client: AsyncRedisClient, *, namespace: str = "modelcore:") -> None:
        if not isinstance(namespace, str):
            raise TypeError("namespace must be a string")
        self._client = client
        self._namespace = namespace

    async def get(self, key: str) -> ChatResponse | None:
        try:
            payload = await self._client.get(self._key(key))
        except (RedisConnectionError, RedisTimeoutError) as error:
            raise CacheUnavailableError("Redis cache is unavailable") from error
        except RedisError as error:
            raise CacheBackendError("Redis cache get failed") from error

        if payload is None:
            return None
        return _deserialize_response(payload)

    async def set(self, key: str, value: ChatResponse, ttl: float | None = None) -> None:
        if ttl is not None and ttl < 0:
            raise ValueError("ttl cannot be negative")

        namespaced_key = self._key(key)
        try:
            if ttl == 0:
                await self._client.delete(namespaced_key)
                return

            payload = _serialize_response(value)
            if ttl is None:
                await self._client.set(namespaced_key, payload)
            else:
                await self._client.set(namespaced_key, payload, px=max(1, math.ceil(ttl * 1000)))
        except (RedisConnectionError, RedisTimeoutError) as error:
            raise CacheUnavailableError("Redis cache is unavailable") from error
        except RedisError as error:
            raise CacheBackendError("Redis cache set failed") from error

    def _key(self, key: str) -> str:
        return f"{self._namespace}{key}"


def _serialize_response(response: ChatResponse) -> bytes:
    usage = response.usage
    payload = {
        "version": _PAYLOAD_VERSION,
        "response": {
            "content": response.content,
            "model": response.model,
            "provider": response.provider,
            "finish_reason": response.finish_reason,
            "usage": None
            if usage is None
            else {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "total_tokens": usage.total_tokens,
            },
        },
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _deserialize_response(payload: bytes | str) -> ChatResponse | None:
    try:
        decoded = json.loads(payload)
        if not isinstance(decoded, dict) or type(decoded.get("version")) is not int:
            return None
        if decoded["version"] != _PAYLOAD_VERSION:
            return None

        response = decoded.get("response")
        if not isinstance(response, dict):
            return None
        content = response.get("content")
        model = response.get("model")
        provider = response.get("provider")
        finish_reason = response.get("finish_reason")
        if not isinstance(content, str) or not isinstance(model, str) or not isinstance(provider, str):
            return None
        if finish_reason is not None and not isinstance(finish_reason, str):
            return None

        usage = _deserialize_usage(response.get("usage"))
        if response.get("usage") is not None and usage is None:
            return None
        return ChatResponse(
            content=content,
            model=model,
            provider=provider,
            usage=usage,
            finish_reason=finish_reason,
        )
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError, ValueError):
        return None


def _deserialize_usage(payload: object) -> Usage | None:
    if payload is None:
        return None
    if not isinstance(payload, dict):
        return None
    input_tokens = payload.get("input_tokens")
    output_tokens = payload.get("output_tokens")
    total_tokens = payload.get("total_tokens")
    if type(input_tokens) is not int or type(output_tokens) is not int:
        return None
    if total_tokens is not None and type(total_tokens) is not int:
        return None
    return Usage(input_tokens=input_tokens, output_tokens=output_tokens, total_tokens=total_tokens)
