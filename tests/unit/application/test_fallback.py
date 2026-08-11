import asyncio

import pytest

from modelcore.application.fallback import FallbackProvider
from modelcore.application.resilience import ResilientProvider, RetryPolicy
from modelcore.exceptions.provider import (
    AuthenticationError,
    GenerationTimeoutError,
    ProviderError,
    ProviderUnavailableError,
    RateLimitError,
)
from modelcore.models.chat_request import ChatRequest
from modelcore.models.chat_response import ChatResponse
from modelcore.models.chat_stream_chunk import ChatStreamChunk
from modelcore.models.message import Message


def make_request() -> ChatRequest:
    return ChatRequest(messages=[Message.user("Hello")], model="test-model")


def make_response(provider: str) -> ChatResponse:
    return ChatResponse(content="Hello", model="test-model", provider=provider, usage=None)


class FakeProvider:
    def __init__(self, name: str, outcomes: list[ChatResponse | BaseException]) -> None:
        self.name = name
        self._outcomes = outcomes
        self.generate_calls = 0
        self.stream_calls = 0

    async def generate(self, request: ChatRequest) -> ChatResponse:
        outcome = self._outcomes[self.generate_calls]
        self.generate_calls += 1
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    async def _stream(self):
        yield ChatStreamChunk(content_delta=self.name, model="test-model", provider=self.name)

    def stream(self, request: ChatRequest):
        self.stream_calls += 1
        return self._stream()


@pytest.mark.asyncio
async def test_primary_success_does_not_call_fallback_and_preserves_response() -> None:
    primary = FakeProvider("primary", [make_response("openai")])
    fallback = FakeProvider("fallback", [make_response("ollama")])

    response = await FallbackProvider([primary, fallback]).generate(make_request())

    assert response.provider == "openai"
    assert primary.generate_calls == 1
    assert fallback.generate_calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error", [ProviderUnavailableError("down"), RateLimitError("limited"), GenerationTimeoutError("slow")]
)
async def test_eligible_primary_errors_try_the_next_provider(error: ProviderError) -> None:
    primary = FakeProvider("primary", [error])
    fallback = FakeProvider("fallback", [make_response("ollama")])

    response = await FallbackProvider([primary, fallback]).generate(make_request())

    assert response.provider == "ollama"
    assert primary.generate_calls == fallback.generate_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [AuthenticationError("bad key"), ProviderError("bad request")])
async def test_non_eligible_modelcore_errors_stop_the_chain(error: ProviderError) -> None:
    primary = FakeProvider("primary", [error])
    fallback = FakeProvider("fallback", [make_response("ollama")])

    with pytest.raises(type(error)):
        await FallbackProvider([primary, fallback]).generate(make_request())

    assert primary.generate_calls == 1
    assert fallback.generate_calls == 0


@pytest.mark.asyncio
async def test_cancellation_propagates_immediately_without_fallback() -> None:
    primary = FakeProvider("primary", [asyncio.CancelledError()])
    fallback = FakeProvider("fallback", [make_response("ollama")])

    with pytest.raises(asyncio.CancelledError):
        await FallbackProvider([primary, fallback]).generate(make_request())

    assert primary.generate_calls == 1
    assert fallback.generate_calls == 0


@pytest.mark.asyncio
async def test_programming_errors_propagate_without_fallback() -> None:
    primary = FakeProvider("primary", [RuntimeError("bug")])
    fallback = FakeProvider("fallback", [make_response("ollama")])

    with pytest.raises(RuntimeError, match="bug"):
        await FallbackProvider([primary, fallback]).generate(make_request())

    assert primary.generate_calls == 1
    assert fallback.generate_calls == 0


@pytest.mark.asyncio
async def test_tries_providers_in_order_until_a_later_provider_succeeds() -> None:
    first = FakeProvider("first", [ProviderUnavailableError("down")])
    second = FakeProvider("second", [RateLimitError("limited")])
    third = FakeProvider("third", [make_response("ollama")])

    response = await FallbackProvider([first, second, third]).generate(make_request())

    assert response.provider == "ollama"
    assert [first.generate_calls, second.generate_calls, third.generate_calls] == [1, 1, 1]


@pytest.mark.asyncio
async def test_last_eligible_error_is_propagated_when_all_providers_fail() -> None:
    first = FakeProvider("first", [ProviderUnavailableError("down")])
    last_error = GenerationTimeoutError("slow")
    second = FakeProvider("second", [last_error])

    with pytest.raises(GenerationTimeoutError) as raised:
        await FallbackProvider([first, second]).generate(make_request())

    assert raised.value is last_error
    assert first.generate_calls == second.generate_calls == 1


@pytest.mark.asyncio
async def test_stream_delegates_only_to_primary_provider() -> None:
    primary = FakeProvider("primary", [make_response("openai")])
    fallback = FakeProvider("fallback", [make_response("ollama")])

    chunks = [chunk async for chunk in FallbackProvider([primary, fallback]).stream(make_request())]

    assert [chunk.content_delta for chunk in chunks] == ["primary"]
    assert primary.stream_calls == 1
    assert fallback.stream_calls == 0


@pytest.mark.asyncio
async def test_each_request_starts_again_with_primary_provider() -> None:
    primary = FakeProvider("primary", [ProviderUnavailableError("down"), make_response("openai")])
    fallback = FakeProvider("fallback", [make_response("ollama")])
    provider = FallbackProvider([primary, fallback])

    first = await provider.generate(make_request())
    second = await provider.generate(make_request())

    assert first.provider == "ollama"
    assert second.provider == "openai"
    assert primary.generate_calls == 2
    assert fallback.generate_calls == 1


@pytest.mark.asyncio
async def test_resilience_retries_one_provider_before_fallback() -> None:
    primary = FakeProvider("primary", [ProviderUnavailableError("once"), ProviderUnavailableError("twice")])
    fallback = FakeProvider("fallback", [make_response("ollama")])
    resilient_primary = ResilientProvider(
        primary,
        RetryPolicy(max_attempts=2, base_delay=0),
        sleep=lambda _: asyncio.sleep(0),
    )

    response = await FallbackProvider([resilient_primary, fallback]).generate(make_request())

    assert response.provider == "ollama"
    assert primary.generate_calls == 2
    assert fallback.generate_calls == 1


def test_fallback_provider_requires_non_empty_unique_provider_sequence() -> None:
    provider = FakeProvider("primary", [make_response("openai")])

    with pytest.raises(ValueError, match="at least one provider"):
        FallbackProvider([])
    with pytest.raises(ValueError, match="must not repeat"):
        FallbackProvider([provider, provider])
