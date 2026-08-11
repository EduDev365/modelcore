from dataclasses import dataclass

_SUPPORTED_ROLES = frozenset({"system", "user", "assistant"})


@dataclass(frozen=True, slots=True)
class Message:
    """A provider-independent message exchanged with a chat model."""

    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in _SUPPORTED_ROLES:
            raise ValueError(f"Unsupported message role: {self.role!r}")
        if not isinstance(self.content, str):
            raise TypeError("Message content must be a string")

    @classmethod
    def system(cls, content: str) -> "Message":
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> "Message":
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str) -> "Message":
        return cls(role="assistant", content=content)
