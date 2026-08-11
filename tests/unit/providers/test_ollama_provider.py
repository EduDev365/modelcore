from types import SimpleNamespace

import pytest
from ollama import ResponseError

from modelcore.config.ollama import OllamaConfig
from modelcore.exceptions.provider import ProviderError, ProviderUnavailableError
from modelcore.models.chat_request import ChatRequest
from modelcore.models.message import Message
from modelcore.providers.ollama import OllamaProvider


class FakeOllamaClient:
    def __init__(self, response: object | None = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.received: dict[str, object] | None = None

    async def chat(self, **kwargs: object) -> object:
        self.received = kwargs
        if self.error is not None:
            raise self.error
        return self.response


def make_response() -> SimpleNamespace:
    return SimpleNamespace(
        model="llama3.2",
        message=SimpleNamespace(content="Answer"),
        done_reason="stop",
        prompt_eval_count=8,
        eval_count=3,
    )


@pytest.mark.asyncio
async def test_ollama_provider_converts_request_and_normalizes_response() -> None:
    client = FakeOllamaClient(response=make_response())
    provider = OllamaProvider(client=client, config=OllamaConfig())
    request = ChatRequest(
        messages=[
            Message.system("You are helpful."),
            Message.user("Explain this."),
            Message.assistant("I will help."),
        ],
        model="llama3.2",
        temperature=0.4,
    )

    response = await provider.generate(request)

    assert client.received == {
        "model": "llama3.2",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Explain this."},
            {"role": "assistant", "content": "I will help."},
        ],
        "options": {"temperature": 0.4},
        "stream": False,
    }
    assert response.content == "Answer"
    assert response.model == "llama3.2"
    assert response.provider == "ollama"
    assert response.usage is not None
    assert response.usage.input_tokens == 8
    assert response.usage.output_tokens == 3
    assert response.usage.total_tokens == 11
    assert response.finish_reason == "stop"


@pytest.mark.asyncio
async def test_ollama_provider_preserves_missing_usage() -> None:
    response_without_usage = SimpleNamespace(
        model="llama3.2",
        message=SimpleNamespace(content="Answer"),
        done_reason=None,
    )
    provider = OllamaProvider(client=FakeOllamaClient(response=response_without_usage), config=OllamaConfig())

    response = await provider.generate(ChatRequest(messages=[Message.user("Hello")], model="llama3.2"))

    assert response.usage is None


@pytest.mark.asyncio
async def test_ollama_provider_maps_response_errors() -> None:
    provider = OllamaProvider(
        client=FakeOllamaClient(error=ResponseError("model unavailable", status_code=404)),
        config=OllamaConfig(),
    )

    with pytest.raises(ProviderError, match="Ollama request failed"):
        await provider.generate(ChatRequest(messages=[Message.user("Hello")], model="llama3.2"))


@pytest.mark.asyncio
async def test_ollama_provider_maps_server_response_errors_to_unavailability() -> None:
    provider = OllamaProvider(
        client=FakeOllamaClient(error=ResponseError("server unavailable", status_code=503)),
        config=OllamaConfig(),
    )

    with pytest.raises(ProviderUnavailableError, match="Ollama provider unavailable"):
        await provider.generate(ChatRequest(messages=[Message.user("Hello")], model="llama3.2"))


@pytest.mark.asyncio
async def test_ollama_provider_does_not_mask_unexpected_errors() -> None:
    provider = OllamaProvider(
        client=FakeOllamaClient(error=RuntimeError("bug")),
        config=OllamaConfig(),
    )

    with pytest.raises(RuntimeError, match="bug"):
        await provider.generate(ChatRequest(messages=[Message.user("Hello")], model="llama3.2"))
