from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real

from modelcore.models.embedding_usage import EmbeddingUsage


@dataclass(frozen=True, slots=True)
class EmbeddingResponse:
    """Provider-independent normalized embedding result."""

    embeddings: tuple[tuple[float, ...], ...]
    model: str
    provider: str
    usage: EmbeddingUsage | None = None

    def __init__(
        self,
        embeddings: Sequence[Sequence[float]],
        model: str,
        provider: str,
        usage: EmbeddingUsage | None = None,
    ) -> None:
        normalized_embeddings = tuple(self._normalize_vector(vector) for vector in embeddings)
        if not normalized_embeddings:
            raise ValueError("EmbeddingResponse requires at least one vector")
        dimensions = {len(vector) for vector in normalized_embeddings}
        if len(dimensions) != 1:
            raise ValueError("EmbeddingResponse vectors must have the same dimension")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("EmbeddingResponse model cannot be blank")
        if not isinstance(provider, str) or not provider.strip():
            raise ValueError("EmbeddingResponse provider cannot be blank")
        if usage is not None and not isinstance(usage, EmbeddingUsage):
            raise TypeError("EmbeddingResponse usage must be an EmbeddingUsage instance")

        object.__setattr__(self, "embeddings", normalized_embeddings)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "usage", usage)

    @staticmethod
    def _normalize_vector(vector: Sequence[float]) -> tuple[float, ...]:
        if not vector:
            raise ValueError("EmbeddingResponse vectors cannot be empty")
        if not all(isinstance(value, Real) and not isinstance(value, bool) for value in vector):
            raise TypeError("EmbeddingResponse vector values must be numbers")
        return tuple(float(value) for value in vector)
