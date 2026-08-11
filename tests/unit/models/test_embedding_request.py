import pytest

from modelcore.models.embedding_request import EmbeddingRequest


def test_embedding_request_normalizes_texts_to_a_tuple() -> None:
    request = EmbeddingRequest(texts=["one", "two"], model="embedding-model")

    assert request.texts == ("one", "two")
    assert request.model == "embedding-model"


def test_embedding_request_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one text"):
        EmbeddingRequest(texts=[], model="embedding-model")


def test_embedding_request_rejects_blank_text() -> None:
    with pytest.raises(ValueError, match="cannot be blank"):
        EmbeddingRequest(texts=[" "], model="embedding-model")


def test_embedding_request_rejects_a_single_string_instead_of_a_sequence() -> None:
    with pytest.raises(TypeError, match="sequence"):
        EmbeddingRequest(texts="text", model="embedding-model")  # type: ignore[arg-type]
