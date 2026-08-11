import pytest

from modelcore.models.embedding_response import EmbeddingResponse
from modelcore.models.embedding_usage import EmbeddingUsage


def test_embedding_response_preserves_normalized_vectors_and_metadata() -> None:
    response = EmbeddingResponse(
        embeddings=[[0.1, 0.2], [0.3, 0.4]],
        model="embedding-model",
        provider="example-provider",
        usage=EmbeddingUsage(input_tokens=4),
    )

    assert response.embeddings == ((0.1, 0.2), (0.3, 0.4))
    assert response.usage is not None
    assert response.usage.total_tokens is None


def test_embedding_response_rejects_inconsistent_vector_dimensions() -> None:
    with pytest.raises(ValueError, match="same dimension"):
        EmbeddingResponse(
            embeddings=[[0.1, 0.2], [0.3]],
            model="embedding-model",
            provider="example-provider",
        )


def test_embedding_response_rejects_empty_vectors() -> None:
    with pytest.raises(ValueError, match="at least one vector"):
        EmbeddingResponse(embeddings=[], model="embedding-model", provider="example-provider")


def test_embedding_usage_rejects_inconsistent_total_tokens() -> None:
    with pytest.raises(ValueError, match="total_tokens must equal"):
        EmbeddingUsage(input_tokens=4, total_tokens=5)
