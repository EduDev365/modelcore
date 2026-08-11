import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable

import pytest

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
from modelcore.models.usage import Usage


class FakeProvider:
    def __init__(self, outcomes: list[ChatResponse | BaseException]) -> None:
        self._outcomes = outcomes
        self.calls = 0

    async def generate(self, request: ChatRequest) -> ChatResponse:
        self.calls += 1
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    async def _empty_stream(self) -> AsyncIterator[ChatStreamChunk]:
        if False:
            yield

    def stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        return self._empty_stream()


class TimeoutThenSuccessProvider:
    def __init__(self, response: ChatResponse) -> None:
        self.calls = 0
        self._response = response

    async def generate(self, request: ChatRequest) -> ChatResponse:
        self.calls += 1
        if self.calls == 1:
            await asyncio.Future()
        return self._response

    def stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        return FakeProvider([]).stream(request)


def make_request() -> ChatRequest:
    return ChatRequest(messages=[Message.user("Hello")], model="example-model")


def make_response() -> ChatResponse:
    return ChatResponse(
        content="Answer",
        model="example-model",
        provider="fake",
        usage=Usage(input_tokens=1, output_tokens=1),
    )


def recording_sleep(delays: list[float]) -> Callable[[float], Awaitable[None]]:
    async def sleep(delay: float) -> None:
        delays.append(delay)

    return sleep


def test_retry_policy_validates_configuration() -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        RetryPolicy(max_attempts=0)
    with pytest.raises(ValueError, match="base_delay"):
        RetryPolicy(base_delay=-1)
    with pytest.raises(ValueError, match="max_delay"):
        RetryPolicy(base_delay=1, max_delay=0.5)


def test_retry_policy_calculates_capped_exponential_backoff() -> None:
    policy = RetryPolicy(max_attempts=4, base_delay=0.25, max_delay=0.75)

    assert [policy.delay_for_retry(attempt) for attempt in (1, 2, 3)] == [0.25, 0.5, 0.75]


def test_retry_policy_classifies_only_transient_modelcore_errors_as_retryable() -> None:
    policy = RetryPolicy()

    assert policy.is_retryable(RateLimitError("limited"))
    assert policy.is_retryable(ProviderUnavailableError("down"))
    assert policy.is_retryable(GenerationTimeoutError("timed out"))
    assert not policy.is_retryable(AuthenticationError("invalid credentials"))
    assert not policy.is_retryable(ProviderError("invalid request"))


@pytest.mark.asyncio
async def test_resilient_provider_returns_on_first_success_without_sleep() -> None:
    provider = FakeProvider([make_response()])
    delays: list[float] = []
    resilient = ResilientProvider(provider, RetryPolicy(), timeout=0.1, sleep=recording_sleep(delays))

    response = await resilient.generate(make_request())

    assert response.content == "Answer"
    assert provider.calls == 1
    assert delays == []


@pytest.mark.asyncio
async def test_resilient_provider_retries_rate_limit_and_returns_success() -> None:
    provider = FakeProvider([RateLimitError("limited"), make_response()])
    delays: list[float] = []
    resilient = ResilientProvider(provider, RetryPolicy(max_attempts=3, base_delay=0.25), sleep=recording_sleep(delays))

    response = await resilient.generate(make_request())

    assert response.content == "Answer"
    assert provider.calls == 2
    assert delays == [0.25]


@pytest.mark.asyncio
async def test_resilient_provider_retries_provider_unavailability_with_exponential_backoff() -> None:
    provider = FakeProvider(
        [
            ProviderUnavailableError("down"),
            ProviderUnavailableError("down"),
            ProviderUnavailableError("down"),
            make_response(),
        ]
    )
    delays: list[float] = []
    resilient = ResilientProvider(
        provider,
        RetryPolicy(max_attempts=4, base_delay=0.25, max_delay=0.75),
        sleep=recording_sleep(delays),
    )

    await resilient.generate(make_request())

    assert provider.calls == 4
    assert delays == [0.25, 0.5, 0.75]


@pytest.mark.asyncio
async def test_resilient_provider_does_not_retry_authentication_errors() -> None:
    provider = FakeProvider([AuthenticationError("invalid credentials")])
    delays: list[float] = []
    resilient = ResilientProvider(provider, RetryPolicy(), sleep=recording_sleep(delays))

    with pytest.raises(AuthenticationError):
        await resilient.generate(make_request())

    assert provider.calls == 1
    assert delays == []


@pytest.mark.asyncio
async def test_resilient_provider_does_not_retry_generic_provider_errors() -> None:
    provider = FakeProvider([ProviderError("invalid response")])
    resilient = ResilientProvider(provider, RetryPolicy())

    with pytest.raises(ProviderError):
        await resilient.generate(make_request())

    assert provider.calls == 1


@pytest.mark.asyncio
async def test_resilient_provider_propagates_final_retryable_error() -> None:
    provider = FakeProvider([RateLimitError("limited"), RateLimitError("limited")])
    delays: list[float] = []
    resilient = ResilientProvider(provider, RetryPolicy(max_attempts=2, base_delay=0.25), sleep=recording_sleep(delays))

    with pytest.raises(RateLimitError):
        await resilient.generate(make_request())

    assert provider.calls == 2
    assert delays == [0.25]


@pytest.mark.asyncio
async def test_resilient_provider_normalizes_timeout_and_retries_it() -> None:
    provider = TimeoutThenSuccessProvider(make_response())
    delays: list[float] = []
    resilient = ResilientProvider(
        provider,
        RetryPolicy(max_attempts=2, base_delay=0),
        timeout=0.001,
        sleep=recording_sleep(delays),
    )

    response = await resilient.generate(make_request())

    assert response.content == "Answer"
    assert provider.calls == 2
    assert delays == [0]


@pytest.mark.asyncio
async def test_resilient_provider_propagates_timeout_when_attempts_are_exhausted() -> None:
    provider = TimeoutThenSuccessProvider(make_response())
    resilient = ResilientProvider(provider, RetryPolicy(max_attempts=1), timeout=0.001)

    with pytest.raises(GenerationTimeoutError):
        await resilient.generate(make_request())


@pytest.mark.asyncio
async def test_resilient_provider_does_not_retry_cancellation() -> None:
    provider = FakeProvider([asyncio.CancelledError()])
    delays: list[float] = []
    resilient = ResilientProvider(provider, RetryPolicy(), sleep=recording_sleep(delays))

    with pytest.raises(asyncio.CancelledError):
        await resilient.generate(make_request())

    assert provider.calls == 1
    assert delays == []


@pytest.mark.asyncio
async def test_resilient_provider_does_not_retry_streams_after_partial_output() -> None:
    class PartialStreamProvider:
        async def generate(self, request: ChatRequest) -> ChatResponse:
            return make_response()

        async def _stream(self) -> AsyncIterator[ChatStreamChunk]:
            yield ChatStreamChunk(content_delta="partial", model="example-model", provider="fake")
            raise RateLimitError("limited")

        def stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
            return self._stream()

    resilient = ResilientProvider(PartialStreamProvider(), RetryPolicy())
    stream = resilient.stream(make_request())

    assert (await anext(stream)).content_delta == "partial"
    with pytest.raises(RateLimitError):
        await anext(stream)
