from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OllamaConfig:
    """Configuration for an Ollama server."""

    base_url: str = "http://localhost:11434"
    timeout: float | None = None

    def __post_init__(self) -> None:
        if not self.base_url.strip():
            raise ValueError("Ollama base_url cannot be blank")
        if self.timeout is not None and self.timeout <= 0:
            raise ValueError("Ollama timeout must be positive")
