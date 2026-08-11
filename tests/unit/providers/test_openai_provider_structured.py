from types import SimpleNamespace

import pytest

from modelcore.config.openai import OpenAIConfig
from modelcore.exceptions.provider import RateLimitError
from modelcore.models.chat_request import ChatRequest
from modelcore.models.message import Message
from modelcore.providers.openai import OpenAIProvider


class FakeCompletions:
    def __init__(self, response: object | None = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.received: dict[str, object] | None = None

    async def create(self, **kwargs: object) -> object:
        self.received = kwargs
        if self.error is not None:
            raise self.error
        return self.response


class FakeClient:
    def __init__(self, completions: FakeCompletions) -> None:
        self.chat = SimpleNamespace(completions=completions)


def make_response() -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"name":"Ada"}'), finish_reason="stop")],
        model="gpt-test",
        usage=SimpleNamespace(prompt_tokens=4, completion_tokens=2, total_tokens=6),
    )


@pytest.mark.asyncio
async def test_openai_provider_sends_strict_schema_and_normalizes_response() -> None:
    completions = FakeCompletions(response=make_response())
    provider = OpenAIProvider(client=FakeClient(completions), config=OpenAIConfig(api_key="fake"))
    request = ChatRequest(messages=[Message.user("Create a person")], model="gpt-test", temperature=0.2)
    schema = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}

    response = await provider.generate_structured(request, schema)

    assert completions.received == {
        "model": "gpt-test",
        "messages": [{"role": "user", "content": "Create a person"}],
        "temperature": 0.2,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "structured_output", "schema": schema, "strict": True},
        },
    }
    assert response.content == '{"name":"Ada"}'
    assert response.provider == "openai"
    assert response.usage is not None
    assert response.usage.total_tokens == 6


@pytest.mark.asyncio
async def test_openai_provider_structured_maps_known_errors() -> None:
    sdk_error = type("RateLimitError", (Exception,), {})("limited")
    provider = OpenAIProvider(
        client=FakeClient(FakeCompletions(error=sdk_error)),
        config=OpenAIConfig(api_key="fake"),
    )

    with pytest.raises(RateLimitError):
        await provider.generate_structured(
            ChatRequest(messages=[Message.user("Create a person")], model="gpt-test"),
            {"type": "object"},
        )


@pytest.mark.asyncio
async def test_openai_provider_structured_does_not_mask_unexpected_errors() -> None:
    provider = OpenAIProvider(
        client=FakeClient(FakeCompletions(error=RuntimeError("bug"))),
        config=OpenAIConfig(api_key="fake"),
    )

    with pytest.raises(RuntimeError, match="bug"):
        await provider.generate_structured(
            ChatRequest(messages=[Message.user("Create a person")], model="gpt-test"),
            {"type": "object"},
        )
