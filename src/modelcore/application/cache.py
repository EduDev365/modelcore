import asyncio
import hashlib
import json
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass

from modelcore.interfaces.cache_backend import CacheBackend
from modelcore.interfaces.llm_provider import LLMProvider
from modelcore.models.chat_request import ChatRequest
from modelcore.models.chat_response import ChatResponse
from modelcore.models.chat_stream_chunk import ChatStreamChunk

Clock = Callable[[], float]


@dataclass(frozen=True, slots=True)
class _CacheEntry:
    value: ChatResponse
    expires_at: float | None


class MemoryCache:
    """Process-local cache with lazy expiry and an injectable monotonic clock."""

    def __init__(self, clock: Clock = time.monotonic) -> None:
        self._clock = clock
        self._entries: dict[str, _CacheEntry] = {}

    async def get(self, key: str) -> ChatResponse | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        if entry.expires_at is not None and self._clock() >= entry.expires_at:
            del self._entries[key]
            return None
        return entry.value

    async def set(self, key: str, value: ChatResponse, ttl: float | None = None) -> None:
        if ttl is not None and ttl < 0:
            raise ValueError("ttl cannot be negative")
        expires_at = None if ttl is None else self._clock() + ttl
        self._entries[key] = _CacheEntry(value=value, expires_at=expires_at)


class CachingProvider:
    """Adds cache lookup and process-local single-flight behavior to a provider."""

    def __init__(
        self,
        provider: LLMProvider,
        cache: CacheBackend,
        *,
        provider_key: str,
        ttl: float | None = None,
    ) -> None:
        if not provider_key.strip():
            raise ValueError("provider_key cannot be blank")
        if ttl is not None and ttl < 0:
            raise ValueError("ttl cannot be negative")
        self._provider = provider
        self._cache = cache
        self._provider_key = provider_key
        self._ttl = ttl
        self._locks: dict[str, asyncio.Lock] = {}

    async def generate(self, request: ChatRequest) -> ChatResponse:
        key = build_cache_key(self._provider_key, request)
        cached_response = await self._cache.get(key)
        if cached_response is not None:
            return cached_response

        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            cached_response = await self._cache.get(key)
            if cached_response is not None:
                return cached_response
            response = await self._provider.generate(request)
            await self._cache.set(key, response, ttl=self._ttl)
            return response

    def stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        """Delegate streaming unchanged; streams are not cached."""
        return self._provider.stream(request)


def build_cache_key(provider_key: str, request: ChatRequest) -> str:
    """Return a stable digest for the fields that affect normal generation."""
    payload = {
        "version": 1,
        "provider": provider_key,
        "model": request.model,
        "temperature": request.temperature,
        "messages": [{"role": message.role, "content": message.content} for message in request.messages],
    }
    canonical_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
