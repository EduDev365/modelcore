from collections.abc import Sequence
from dataclasses import dataclass

from modelcore.models.message import Message


@dataclass(frozen=True, slots=True)
class ChatRequest:
    """Input for a provider-independent chat generation request."""

    messages: tuple[Message, ...]
    model: str
    temperature: float = 1.0

    def __init__(
        self,
        messages: Sequence[Message],
        model: str,
        temperature: float = 1.0,
    ) -> None:
        normalized_messages = tuple(messages)
        if not normalized_messages:
            raise ValueError("ChatRequest requires at least one message")
        if not all(isinstance(message, Message) for message in normalized_messages):
            raise TypeError("ChatRequest messages must be Message instances")
        if not model.strip():
            raise ValueError("model cannot be blank")
        object.__setattr__(self, "messages", normalized_messages)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "temperature", temperature)
