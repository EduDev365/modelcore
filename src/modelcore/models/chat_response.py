from dataclasses import dataclass

from modelcore.models.usage import Usage


@dataclass(frozen=True, slots=True)
class ChatResponse:
    """Provider-independent result of a chat generation request."""

    content: str
    model: str
    provider: str
    usage: Usage | None
    finish_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.content, str):
            raise TypeError("ChatResponse content must be a string")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("ChatResponse model cannot be blank")
        if not isinstance(self.provider, str) or not self.provider.strip():
            raise ValueError("ChatResponse provider cannot be blank")
        if self.usage is not None and not isinstance(self.usage, Usage):
            raise TypeError("ChatResponse usage must be a Usage instance")
