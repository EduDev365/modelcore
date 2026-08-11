import asyncio

import pytest

from modelcore.application.cache import CachingProvider, MemoryCache, build_cache_key
from modelcore.application.resilience import ResilientProvider, RetryPolicy
from modelcore.models.chat_request import ChatRequest
from modelcore.models.chat_response import ChatResponse
from modelcore.models.chat_stream_chunk import ChatStreamChunk
from modelcore.models.message import Message
from modelcore.models.usage import Usage


class FakeClock:
    def __init__(self, value: float = 100.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


def make_request(
    *,
    model: str = "gpt-test",
    temperature: float = 0.2,
    content: str = "Hello",
) -> ChatRequest:
    return ChatRequest(messages=[Message.user(content)], model=model, temperature=temperature)


def make_response(content: str = "Hi") -> ChatResponse:
    return ChatResponse(
        content=content,
        model="gpt-test",
        provider="fake",
        usage=Usage(input_tokens=3, output_tokens=2),
        finish_reason="stop",
    )


class FakeProvider:
    def __init__(self, response: ChatResponse | None = None, error: BaseException | None = None) -> None:
        self.response = response or make_response()
        self.error = error
        self.generate_calls = 0
        self.stream_calls = 0

    async def generate(self, request: ChatRequest) -> ChatResponse:
        self.generate_calls += 1
        if self.error is not None:
            raise self.error
        return self.response

    async def _stream(self):
        yield ChatStreamChunk(content_delta="Hi", model="gpt-test", provider="fake")

    def stream(self, request: ChatRequest):
        self.stream_calls += 1
        return self._stream()


@pytest.mark.asyncio
async def test_memory_cache_returns_missing_and_non_expiring_values() -> None:
    cache = MemoryCache(clock=FakeClock())
    response = make_response()

    assert await cache.get("missing") is None

    await cache.set("key", response)

    assert await cache.get("key") is response


@pytest.mark.asyncio
async def test_memory_cache_expires_entries_lazily_without_real_sleep() -> None:
    clock = FakeClock()
    cache = MemoryCache(clock=clock)
    response = make_response()

    await cache.set("key", response, ttl=10)
    clock.value = 109.9
    assert await cache.get("key") is response

    clock.value = 110.0
    assert await cache.get("key") is None
    assert await cache.get("key") is None


@pytest.mark.asyncio
async def test_memory_cache_treats_zero_ttl_as_immediately_expired_and_rejects_negative_ttl() -> None:
    cache = MemoryCache(clock=FakeClock())

    await cache.set("zero", make_response(), ttl=0)

    assert await cache.get("zero") is None
    with pytest.raises(ValueError, match="ttl cannot be negative"):
        await cache.set("negative", make_response(), ttl=-0.1)


def test_cache_key_is_deterministic_and_includes_semantic_request_fields() -> None:
    request = make_request()

    key = build_cache_key("openai", request)

    assert key == build_cache_key("openai", make_request())
    assert len(key) == 64
    assert key != build_cache_key("ollama", request)
    assert key != build_cache_key("openai", make_request(model="other"))
    assert key != build_cache_key("openai", make_request(temperature=0.3))
    assert key != build_cache_key("openai", make_request(content="Different"))


@pytest.mark.asyncio
async def test_caching_provider_stores_success_and_returns_hit_without_calling_provider() -> None:
    provider = FakeProvider()
    cached = CachingProvider(provider, MemoryCache(), provider_key="fake", ttl=60)
    request = make_request()

    first = await cached.generate(request)
    second = await cached.generate(request)

    assert first is provider.response
    assert second is provider.response
    assert provider.generate_calls == 1


@pytest.mark.asyncio
async def test_caching_provider_does_not_share_entries_between_provider_keys() -> None:
    cache = MemoryCache()
    first_provider = FakeProvider(make_response("one"))
    second_provider = FakeProvider(make_response("two"))
    request = make_request()

    assert (await CachingProvider(first_provider, cache, provider_key="one").generate(request)).content == "one"
    assert (await CachingProvider(second_provider, cache, provider_key="two").generate(request)).content == "two"
    assert first_provider.generate_calls == second_provider.generate_calls == 1


@pytest.mark.asyncio
async def test_cache_outside_resilience_skips_retry_logic_for_a_cache_hit() -> None:
    provider = FakeProvider()
    resilient = ResilientProvider(provider, RetryPolicy(max_attempts=3))
    cached = CachingProvider(resilient, MemoryCache(), provider_key="fake")
    request = make_request()

    await cached.generate(request)
    await cached.generate(request)

    assert provider.generate_calls == 1


@pytest.mark.asyncio
async def test_caching_provider_does_not_cache_provider_errors_or_cancellation() -> None:
    request = make_request()
    failing = FakeProvider(error=RuntimeError("failure"))
    cached_failing = CachingProvider(failing, MemoryCache(), provider_key="fake")

    for _ in range(2):
        with pytest.raises(RuntimeError, match="failure"):
            await cached_failing.generate(request)
    assert failing.generate_calls == 2

    cancelled = FakeProvider(error=asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await CachingProvider(cancelled, MemoryCache(), provider_key="fake").generate(request)
    assert cancelled.generate_calls == 1


@pytest.mark.asyncio
async def test_caching_provider_delegates_stream_without_buffering_or_caching() -> None:
    provider = FakeProvider()
    cached = CachingProvider(provider, MemoryCache(), provider_key="fake")

    chunks = [chunk async for chunk in cached.stream(make_request())]

    assert [chunk.content_delta for chunk in chunks] == ["Hi"]
    assert provider.stream_calls == 1
    assert provider.generate_calls == 0


@pytest.mark.asyncio
async def test_caching_provider_deduplicates_concurrent_misses_for_the_same_key() -> None:
    class BlockingProvider(FakeProvider):
        def __init__(self) -> None:
            super().__init__()
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def generate(self, request: ChatRequest) -> ChatResponse:
            self.generate_calls += 1
            self.started.set()
            await self.release.wait()
            return self.response

    provider = BlockingProvider()
    cached = CachingProvider(provider, MemoryCache(), provider_key="fake")
    request = make_request()

    first = asyncio.create_task(cached.generate(request))
    await provider.started.wait()
    second = asyncio.create_task(cached.generate(request))
    await asyncio.sleep(0)
    provider.release.set()

    assert await first is provider.response
    assert await second is provider.response
    assert provider.generate_calls == 1


def test_caching_provider_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="provider_key cannot be blank"):
        CachingProvider(FakeProvider(), MemoryCache(), provider_key=" ")
    with pytest.raises(ValueError, match="ttl cannot be negative"):
        CachingProvider(FakeProvider(), MemoryCache(), provider_key="fake", ttl=-1)
