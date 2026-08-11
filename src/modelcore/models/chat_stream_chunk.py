from dataclasses import dataclass

from modelcore.models.usage import Usage


@dataclass(frozen=True, slots=True)
class ChatStreamChunk:
    """A normalized incremental update from a chat generation stream."""

    content_delta: str
    model: str
    provider: str
    finish_reason: str | None = None
    usage: Usage | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.content_delta, str):
            raise TypeError("ChatStreamChunk content_delta must be a string")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("ChatStreamChunk model cannot be blank")
        if not isinstance(self.provider, str) or not self.provider.strip():
            raise ValueError("ChatStreamChunk provider cannot be blank")
        if self.usage is not None and not isinstance(self.usage, Usage):
            raise TypeError("ChatStreamChunk usage must be a Usage instance")
