import asyncio
from types import SimpleNamespace

import pytest
from ollama import ResponseError

from modelcore.config.ollama import OllamaConfig
from modelcore.exceptions.provider import ProviderError
from modelcore.models.chat_request import ChatRequest
from modelcore.models.message import Message
from modelcore.providers.ollama import OllamaProvider


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


class FakeOllamaClient:
    def __init__(self, stream: FakeAsyncStream | None = None, error: Exception | None = None) -> None:
        self._stream = stream
        self._error = error
        self.received: dict[str, object] | None = None

    async def chat(self, **kwargs: object) -> FakeAsyncStream:
        self.received = kwargs
        if self._error is not None:
            raise self._error
        assert self._stream is not None
        return self._stream


def make_chunk(
    content: str,
    done: bool = False,
    done_reason: str | None = None,
    prompt_eval_count: int | None = None,
    eval_count: int | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        model="llama3.2",
        message=SimpleNamespace(content=content),
        done=done,
        done_reason=done_reason,
        prompt_eval_count=prompt_eval_count,
        eval_count=eval_count,
    )


@pytest.mark.asyncio
async def test_ollama_provider_streams_normalized_chunks() -> None:
    client = FakeOllamaClient(
        FakeAsyncStream(
            [
                make_chunk("Hel"),
                make_chunk("lo", done=True, done_reason="stop", prompt_eval_count=8, eval_count=3),
            ]
        )
    )
    provider = OllamaProvider(client=client, config=OllamaConfig())
    request = ChatRequest(
        messages=[Message.system("Be brief."), Message.user("Hello")], model="llama3.2", temperature=0.4
    )

    chunks = [chunk async for chunk in provider.stream(request)]

    assert client.received == {
        "model": "llama3.2",
        "messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hello"},
        ],
        "options": {"temperature": 0.4},
        "stream": True,
    }
    assert [chunk.content_delta for chunk in chunks] == ["Hel", "lo"]
    assert chunks[1].finish_reason == "stop"
    assert chunks[1].usage is not None
    assert chunks[1].usage.total_tokens == 11
    assert all(chunk.provider == "ollama" and chunk.model == "llama3.2" for chunk in chunks)


@pytest.mark.asyncio
async def test_ollama_provider_stream_keeps_usage_absent_when_not_reported() -> None:
    provider = OllamaProvider(
        client=FakeOllamaClient(FakeAsyncStream([make_chunk("Hello", done=True, done_reason="stop")])),
        config=OllamaConfig(),
    )

    chunks = [chunk async for chunk in provider.stream(ChatRequest(messages=[Message.user("Hello")], model="llama3.2"))]

    assert chunks[0].usage is None


@pytest.mark.asyncio
async def test_ollama_provider_stream_maps_response_errors() -> None:
    provider = OllamaProvider(
        client=FakeOllamaClient(error=ResponseError("model unavailable", status_code=404)),
        config=OllamaConfig(),
    )

    with pytest.raises(ProviderError, match="Ollama request failed"):
        async for _ in provider.stream(ChatRequest(messages=[Message.user("Hello")], model="llama3.2")):
            pass


@pytest.mark.asyncio
async def test_ollama_provider_stream_does_not_mask_unexpected_errors() -> None:
    provider = OllamaProvider(
        client=FakeOllamaClient(stream=FakeAsyncStream([], error=RuntimeError("bug"))),
        config=OllamaConfig(),
    )

    with pytest.raises(RuntimeError, match="bug"):
        async for _ in provider.stream(ChatRequest(messages=[Message.user("Hello")], model="llama3.2")):
            pass


@pytest.mark.asyncio
async def test_ollama_provider_stream_does_not_swallow_cancellation() -> None:
    provider = OllamaProvider(
        client=FakeOllamaClient(stream=FakeAsyncStream([], error=asyncio.CancelledError())),
        config=OllamaConfig(),
    )

    with pytest.raises(asyncio.CancelledError):
        async for _ in provider.stream(ChatRequest(messages=[Message.user("Hello")], model="llama3.2")):
            pass
