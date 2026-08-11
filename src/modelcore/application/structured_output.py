from typing import TypeVar

from pydantic import BaseModel, ValidationError

from modelcore.exceptions.provider import StructuredOutputError
from modelcore.interfaces.structured_output_provider import StructuredOutputProvider
from modelcore.models.chat_request import ChatRequest

T = TypeVar("T", bound=BaseModel)


class StructuredGeneration:
    """Generates and locally validates typed structured model output."""

    def __init__(self, provider: StructuredOutputProvider) -> None:
        self._provider = provider

    async def generate(self, request: ChatRequest, response_model: type[T]) -> T:
        schema = _schema_for(response_model)
        response = await self._provider.generate_structured(request, schema)
        return parse_structured_output(response.content, response_model)


def parse_structured_output(content: str, response_model: type[T]) -> T:
    _schema_for(response_model)
    try:
        return response_model.model_validate_json(content)
    except (ValidationError, ValueError) as error:
        raise StructuredOutputError("Structured output did not match the requested schema") from error


def _schema_for(response_model: type[T]) -> dict[str, object]:
    if not isinstance(response_model, type) or not issubclass(response_model, BaseModel):
        raise StructuredOutputError("response_model must be a Pydantic BaseModel subclass")
    return response_model.model_json_schema()
