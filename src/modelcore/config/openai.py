from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class OpenAIConfig:
    """Configuration required to construct an OpenAI client."""

    api_key: str = field(repr=False)
    timeout: float | None = None

    def __post_init__(self) -> None:
        if not self.api_key.strip():
            raise ValueError("OpenAI api_key cannot be blank")
        if self.timeout is not None and self.timeout <= 0:
            raise ValueError("OpenAI timeout must be positive")
