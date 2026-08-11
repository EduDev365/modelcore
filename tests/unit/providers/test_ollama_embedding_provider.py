from types import SimpleNamespace

import pytest
from ollama import ResponseError

from modelcore.config.ollama import OllamaConfig
from modelcore.exceptions.provider import ProviderError
from modelcore.models.embedding_request import EmbeddingRequest
from modelcore.providers.ollama_embeddings import OllamaEmbeddingProvider


class FakeOllamaClient:
    def __init__(self, response: object | None = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.received: dict[str, object] | None = None

    async def embed(self, **kwargs: object) -> object:
        self.received = kwargs
        if self.error is not None:
            raise self.error
        return self.response


@pytest.mark.asyncio
async def test_ollama_embedding_provider_normalizes_batch_embeddings() -> None:
    raw_response = SimpleNamespace(
        embeddings=[[0.1, 0.2], [0.3, 0.4]],
        model="embeddinggemma",
        prompt_eval_count=4,
    )
    client = FakeOllamaClient(response=raw_response)
    provider = OllamaEmbeddingProvider(client=client, config=OllamaConfig())

    response = await provider.embed(EmbeddingRequest(texts=["first", "second"], model="embeddinggemma"))

    assert client.received == {"model": "embeddinggemma", "input": ["first", "second"]}
    assert response.embeddings == ((0.1, 0.2), (0.3, 0.4))
    assert response.model == "embeddinggemma"
    assert response.provider == "ollama"
    assert response.usage is not None
    assert response.usage.input_tokens == 4
    assert response.usage.total_tokens is None


@pytest.mark.asyncio
async def test_ollama_embedding_provider_preserves_missing_usage() -> None:
    raw_response = SimpleNamespace(embeddings=[[0.1, 0.2]], model="embeddinggemma")
    provider = OllamaEmbeddingProvider(client=FakeOllamaClient(response=raw_response), config=OllamaConfig())

    response = await provider.embed(EmbeddingRequest(texts=["text"], model="embeddinggemma"))

    assert response.usage is None


@pytest.mark.asyncio
async def test_ollama_embedding_provider_rejects_incomplete_response() -> None:
    raw_response = SimpleNamespace(embeddings=[[0.1, 0.2]], model="embeddinggemma")
    provider = OllamaEmbeddingProvider(client=FakeOllamaClient(response=raw_response), config=OllamaConfig())

    with pytest.raises(ProviderError, match="invalid embedding response"):
        await provider.embed(EmbeddingRequest(texts=["first", "second"], model="embeddinggemma"))


@pytest.mark.asyncio
async def test_ollama_embedding_provider_maps_response_errors() -> None:
    provider = OllamaEmbeddingProvider(
        client=FakeOllamaClient(error=ResponseError("model unavailable", status_code=404)),
        config=OllamaConfig(),
    )

    with pytest.raises(ProviderError, match="Ollama request failed"):
        await provider.embed(EmbeddingRequest(texts=["text"], model="embeddinggemma"))


@pytest.mark.asyncio
async def test_ollama_embedding_provider_does_not_mask_unexpected_errors() -> None:
    provider = OllamaEmbeddingProvider(
        client=FakeOllamaClient(error=RuntimeError("bug")),
        config=OllamaConfig(),
    )

    with pytest.raises(RuntimeError, match="bug"):
        await provider.embed(EmbeddingRequest(texts=["text"], model="embeddinggemma"))
