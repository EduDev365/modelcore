import pytest
from pydantic import BaseModel, ConfigDict

from modelcore.application.structured_output import StructuredGeneration, parse_structured_output
from modelcore.exceptions.provider import StructuredOutputError
from modelcore.models.chat_request import ChatRequest
from modelcore.models.chat_response import ChatResponse
from modelcore.models.message import Message
from modelcore.models.usage import Usage


class Address(BaseModel):
    city: str


class Person(BaseModel):
    name: str
    age: int
    address: Address
    tags: list[str]


class StrictPerson(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str


def test_parse_structured_output_returns_the_requested_typed_model() -> None:
    result = parse_structured_output(
        '{"name":"Ada","age":36,"address":{"city":"London"},"tags":["engineer"]}',
        Person,
    )

    assert result == Person(name="Ada", age=36, address=Address(city="London"), tags=["engineer"])


@pytest.mark.parametrize(
    "content",
    [
        '{"name":"Ada","address":{"city":"London"},"tags":[]}',
        '{"name":"Ada","age":"old","address":{"city":"London"},"tags":[]}',
        '{"name":"Ada",',
    ],
)
def test_parse_structured_output_normalizes_invalid_content_to_modelcore_error(content: str) -> None:
    with pytest.raises(StructuredOutputError, match="did not match the requested schema"):
        parse_structured_output(content, Person)


def test_parse_structured_output_respects_the_schema_extra_field_policy() -> None:
    with pytest.raises(StructuredOutputError, match="did not match the requested schema"):
        parse_structured_output('{"name":"Ada","unexpected":true}', StrictPerson)


def test_parse_structured_output_rejects_a_non_pydantic_response_model() -> None:
    with pytest.raises(StructuredOutputError, match="Pydantic BaseModel"):
        parse_structured_output('{"name":"Ada"}', dict)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_structured_generation_builds_schema_and_returns_a_typed_value() -> None:
    class FakeProvider:
        async def generate_structured(self, request, schema):  # type: ignore[no-untyped-def]
            assert schema == Person.model_json_schema()
            return ChatResponse(
                content='{"name":"Ada","age":36,"address":{"city":"London"},"tags":[]}',
                model="example-model",
                provider="fake",
                usage=Usage(input_tokens=1, output_tokens=1),
            )

    service = StructuredGeneration(FakeProvider())

    result = await service.generate(
        ChatRequest(messages=[Message.user("Create a person")], model="example-model"), Person
    )

    assert result == Person(name="Ada", age=36, address=Address(city="London"), tags=[])
