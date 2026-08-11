from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from modelcore.models.usage import Usage


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    description: str
    arguments_model: type[BaseModel]
    handler: Callable[..., Any]

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.description.strip():
            raise ValueError("tool name and description cannot be blank")
        if not isinstance(self.arguments_model, type) or not issubclass(self.arguments_model, BaseModel):
            raise TypeError("arguments_model must be a Pydantic BaseModel subclass")
        if not callable(self.handler):
            raise TypeError("handler must be callable")

    @property
    def json_schema(self) -> dict[str, object]:
        return self.arguments_model.model_json_schema()


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str | None
    name: str
    arguments: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("tool call name cannot be blank")
        if not isinstance(self.arguments, dict):
            raise TypeError("tool call arguments must be a dictionary")


@dataclass(frozen=True, slots=True)
class ToolResult:
    tool_call_id: str | None
    name: str
    content: str


@dataclass(frozen=True, slots=True)
class ToolCallingResponse:
    content: str | None
    tool_calls: tuple[ToolCall, ...]
    model: str
    provider: str
    usage: Usage | None
    finish_reason: str | None = None
