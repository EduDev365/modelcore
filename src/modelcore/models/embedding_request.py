from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EmbeddingRequest:
    """Provider-independent input for an embedding operation."""

    texts: tuple[str, ...]
    model: str

    def __init__(self, texts: Sequence[str], model: str) -> None:
        if isinstance(texts, str) or not isinstance(texts, Sequence):
            raise TypeError("texts must be a sequence of strings")
        normalized_texts = tuple(texts)
        if not normalized_texts:
            raise ValueError("EmbeddingRequest requires at least one text")
        if not all(isinstance(text, str) for text in normalized_texts):
            raise TypeError("EmbeddingRequest texts must be strings")
        if any(not text.strip() for text in normalized_texts):
            raise ValueError("EmbeddingRequest text cannot be blank")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model cannot be blank")

        object.__setattr__(self, "texts", normalized_texts)
        object.__setattr__(self, "model", model)
