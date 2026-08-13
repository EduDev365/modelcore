from typing import Protocol

from modelcore.models.embedding_request import EmbeddingRequest
from modelcore.models.embedding_response import EmbeddingResponse


class EmbeddingProvider(Protocol):
    """Generate provider-independent embeddings."""

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generate normalized embeddings for a sequence of texts."""
