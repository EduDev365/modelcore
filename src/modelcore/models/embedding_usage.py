from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EmbeddingUsage:
    """Input token usage reported by an embedding provider."""

    input_tokens: int
    total_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.input_tokens < 0:
            raise ValueError("input_tokens cannot be negative")
        if self.total_tokens is not None and self.total_tokens < 0:
            raise ValueError("total_tokens cannot be negative")
        if self.total_tokens is not None and self.total_tokens != self.input_tokens:
            raise ValueError("total_tokens must equal input_tokens")
