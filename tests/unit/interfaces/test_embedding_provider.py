from typing import get_type_hints

from modelcore.interfaces.embedding_provider import EmbeddingProvider


def test_embedding_provider_declares_only_embedding_capability() -> None:
    annotations = get_type_hints(EmbeddingProvider.embed)

    assert "request" in annotations
    assert "return" in annotations
    assert not hasattr(EmbeddingProvider, "generate")
    assert not hasattr(EmbeddingProvider, "stream")
