import asyncio
from types import SimpleNamespace

import pytest

from modelcore.config.openai import OpenAIConfig
from modelcore.exceptions.provider import AuthenticationError
from modelcore.models.chat_request import ChatRequest
from modelcore.models.message import Message
from modelcore.providers.openai import OpenAIProvider


class FakeAsyncStream:
    def __init__(self, chunks: list[object], error: Exception | None = None) -> None:
        self._chunks = chunks
        self._error = error

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for chunk in self._chunks:
            yield chunk
        if self._error is not None:
            raise self._error


class FakeCompletions:
    def __init__(self, stream: FakeAsyncStream | None = None, error: Exception | None = None) -> None:
        self._stream = stream
        self._error = error
        self.received: dict[str, object] | None = None

    async def create(self, **kwargs: object) -> FakeAsyncStream:
        self.received = kwargs
        if self._error is not None:
            raise self._error
        assert self._stream is not None
        return self._stream


class FakeClient:
    def __init__(self, completions: FakeCompletions) -> None:
        self.chat = SimpleNamespace(completions=completions)


def make_chunk(
    content: str | None = None,
    finish_reason: str | None = None,
    usage: object | None = None,
) -> SimpleNamespace:
    choices = (
        []
        if content is None and finish_reason is None
        else [SimpleNamespace(delta=SimpleNamespace(content=content), finish_reason=finish_reason)]
    )
    return SimpleNamespace(model="gpt-test", choices=choices, usage=usage)


@pytest.mark.asyncio
async def test_openai_provider_streams_normalized_chunks() -> None:
    usage = SimpleNamespace(prompt_tokens=8, completion_tokens=3, total_tokens=11)
    completions = FakeCompletions(
        FakeAsyncStream(
            [
                make_chunk(content="Hel"),
                make_chunk(),
                make_chunk(content="lo"),
                make_chunk(finish_reason="stop"),
                make_chunk(usage=usage),
            ]
        )
    )
    provider = OpenAIProvider(client=FakeClient(completions), config=OpenAIConfig(api_key="fake"))
    request = ChatRequest(
        messages=[Message.system("Be brief."), Message.user("Hello")], model="gpt-test", temperature=0.4
    )

    chunks = [chunk async for chunk in provider.stream(request)]

    assert completions.received == {
        "model": "gpt-test",
        "messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hello"},
        ],
        "temperature": 0.4,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    assert [chunk.content_delta for chunk in chunks] == ["Hel", "lo", "", ""]
    assert chunks[2].finish_reason == "stop"
    assert chunks[3].usage is not None
    assert chunks[3].usage.total_tokens == 11
    assert all(chunk.provider == "openai" and chunk.model == "gpt-test" for chunk in chunks)


@pytest.mark.asyncio
async def test_openai_provider_stream_maps_known_errors() -> None:
    sdk_error = type("AuthenticationError", (Exception,), {})("invalid key")
    provider = OpenAIProvider(
        client=FakeClient(FakeCompletions(error=sdk_error)),
        config=OpenAIConfig(api_key="fake"),
    )

    with pytest.raises(AuthenticationError):
        async for _ in provider.stream(ChatRequest(messages=[Message.user("Hello")], model="gpt-test")):
            pass


@pytest.mark.asyncio
async def test_openai_provider_stream_does_not_mask_unexpected_errors() -> None:
    provider = OpenAIProvider(
        client=FakeClient(FakeCompletions(stream=FakeAsyncStream([], error=RuntimeError("bug")))),
        config=OpenAIConfig(api_key="fake"),
    )

    with pytest.raises(RuntimeError, match="bug"):
        async for _ in provider.stream(ChatRequest(messages=[Message.user("Hello")], model="gpt-test")):
            pass


@pytest.mark.asyncio
async def test_openai_provider_stream_does_not_swallow_cancellation() -> None:
    provider = OpenAIProvider(
        client=FakeClient(FakeCompletions(stream=FakeAsyncStream([], error=asyncio.CancelledError()))),
        config=OpenAIConfig(api_key="fake"),
    )

    with pytest.raises(asyncio.CancelledError):
        async for _ in provider.stream(ChatRequest(messages=[Message.user("Hello")], model="gpt-test")):
            pass
