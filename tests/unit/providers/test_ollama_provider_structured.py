from types import SimpleNamespace

import pytest
from ollama import ResponseError

from modelcore.config.ollama import OllamaConfig
from modelcore.exceptions.provider import ProviderError
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
        message=SimpleNamespace(content='{"name":"Ada"}'),
        done_reason="stop",
        prompt_eval_count=4,
        eval_count=2,
    )


@pytest.mark.asyncio
async def test_ollama_provider_sends_schema_format_and_normalizes_response() -> None:
    client = FakeOllamaClient(response=make_response())
    provider = OllamaProvider(client=client, config=OllamaConfig())
    request = ChatRequest(messages=[Message.user("Create a person")], model="llama3.2", temperature=0.2)
    schema = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}

    response = await provider.generate_structured(request, schema)

    assert client.received == {
        "model": "llama3.2",
        "messages": [{"role": "user", "content": "Create a person"}],
        "options": {"temperature": 0.2},
        "format": schema,
        "stream": False,
    }
    assert response.content == '{"name":"Ada"}'
    assert response.provider == "ollama"
    assert response.usage is not None
    assert response.usage.total_tokens == 6


@pytest.mark.asyncio
async def test_ollama_provider_structured_maps_response_errors() -> None:
    provider = OllamaProvider(
        client=FakeOllamaClient(error=ResponseError("model unavailable", status_code=404)),
        config=OllamaConfig(),
    )

    with pytest.raises(ProviderError, match="Ollama request failed"):
        await provider.generate_structured(
            ChatRequest(messages=[Message.user("Create a person")], model="llama3.2"),
            {"type": "object"},
        )


@pytest.mark.asyncio
async def test_ollama_provider_structured_does_not_mask_unexpected_errors() -> None:
    provider = OllamaProvider(
        client=FakeOllamaClient(error=RuntimeError("bug")),
        config=OllamaConfig(),
    )

    with pytest.raises(RuntimeError, match="bug"):
        await provider.generate_structured(
            ChatRequest(messages=[Message.user("Create a person")], model="llama3.2"),
            {"type": "object"},
        )
