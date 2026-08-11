from types import SimpleNamespace

import pytest

from modelcore.config.openai import OpenAIConfig
from modelcore.exceptions.provider import ProviderError, RateLimitError
from modelcore.models.embedding_request import EmbeddingRequest
from modelcore.providers.openai_embeddings import OpenAIEmbeddingProvider


class FakeEmbeddingsClient:
    def __init__(self, response: object | None = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.received: dict[str, object] | None = None

    async def create(self, **kwargs: object) -> object:
        self.received = kwargs
        if self.error is not None:
            raise self.error
        return self.response


class FakeClient:
    def __init__(self, embeddings: FakeEmbeddingsClient) -> None:
        self.embeddings = embeddings


def make_response() -> SimpleNamespace:
    return SimpleNamespace(
        data=[
            SimpleNamespace(index=1, embedding=[0.3, 0.4]),
            SimpleNamespace(index=0, embedding=[0.1, 0.2]),
        ],
        model="text-embedding-test",
        usage=SimpleNamespace(prompt_tokens=4, total_tokens=4),
    )


@pytest.mark.asyncio
async def test_openai_embedding_provider_normalizes_and_orders_embeddings() -> None:
    embeddings = FakeEmbeddingsClient(response=make_response())
    provider = OpenAIEmbeddingProvider(client=FakeClient(embeddings), config=OpenAIConfig(api_key="fake"))

    response = await provider.embed(EmbeddingRequest(texts=["first", "second"], model="text-embedding-test"))

    assert embeddings.received == {"model": "text-embedding-test", "input": ["first", "second"]}
    assert response.embeddings == ((0.1, 0.2), (0.3, 0.4))
    assert response.model == "text-embedding-test"
    assert response.provider == "openai"
    assert response.usage is not None
    assert response.usage.input_tokens == 4
    assert response.usage.total_tokens == 4


@pytest.mark.asyncio
async def test_openai_embedding_provider_rejects_incomplete_indexed_response() -> None:
    incomplete_response = SimpleNamespace(
        data=[SimpleNamespace(index=1, embedding=[0.3, 0.4])],
        model="text-embedding-test",
        usage=SimpleNamespace(prompt_tokens=2, total_tokens=2),
    )
    provider = OpenAIEmbeddingProvider(
        client=FakeClient(FakeEmbeddingsClient(response=incomplete_response)),
        config=OpenAIConfig(api_key="fake"),
    )

    with pytest.raises(ProviderError, match="invalid embedding response"):
        await provider.embed(EmbeddingRequest(texts=["first", "second"], model="text-embedding-test"))


@pytest.mark.asyncio
async def test_openai_embedding_provider_maps_known_errors() -> None:
    sdk_error = type("RateLimitError", (Exception,), {})("limited")
    provider = OpenAIEmbeddingProvider(
        client=FakeClient(FakeEmbeddingsClient(error=sdk_error)),
        config=OpenAIConfig(api_key="fake"),
    )

    with pytest.raises(RateLimitError):
        await provider.embed(EmbeddingRequest(texts=["text"], model="text-embedding-test"))


@pytest.mark.asyncio
async def test_openai_embedding_provider_does_not_mask_unexpected_errors() -> None:
    provider = OpenAIEmbeddingProvider(
        client=FakeClient(FakeEmbeddingsClient(error=RuntimeError("bug"))),
        config=OpenAIConfig(api_key="fake"),
    )

    with pytest.raises(RuntimeError, match="bug"):
        await provider.embed(EmbeddingRequest(texts=["text"], model="text-embedding-test"))
