from types import SimpleNamespace

import pytest

from modelcore.config.openai import OpenAIConfig
from modelcore.exceptions.provider import AuthenticationError, ProviderUnavailableError, RateLimitError
from modelcore.models.chat_request import ChatRequest
from modelcore.models.message import Message
from modelcore.providers.openai import OpenAIProvider


class FakeCompletions:
    def __init__(self, response=None, error: Exception | None = None) -> None:
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
        choices=[SimpleNamespace(message=SimpleNamespace(content="Answer"), finish_reason="stop")],
        model="returned-model",
        usage=SimpleNamespace(prompt_tokens=8, completion_tokens=3, total_tokens=11),
    )


@pytest.mark.asyncio
async def test_openai_provider_converts_request_and_normalizes_response() -> None:
    completions = FakeCompletions(response=make_response())
    provider = OpenAIProvider(client=FakeClient(completions), config=OpenAIConfig(api_key="fake"))
    request = ChatRequest(
        messages=[
            Message.system("You are helpful."),
            Message.user("Explain this."),
            Message.assistant("I will help."),
        ],
        model="requested-model",
        temperature=0.4,
    )

    response = await provider.generate(request)

    assert completions.received == {
        "model": "requested-model",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Explain this."},
            {"role": "assistant", "content": "I will help."},
        ],
        "temperature": 0.4,
    }
    assert response.content == "Answer"
    assert response.model == "returned-model"
    assert response.provider == "openai"
    assert response.usage.input_tokens == 8
    assert response.usage.output_tokens == 3
    assert response.usage.total_tokens == 11
    assert response.finish_reason == "stop"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error_type", "expected_type"),
    [
        ("AuthenticationError", AuthenticationError),
        ("RateLimitError", RateLimitError),
        ("APIError", ProviderUnavailableError),
    ],
)
async def test_openai_provider_maps_known_sdk_errors(error_type: str, expected_type: type[Exception]) -> None:
    sdk_error = type(error_type, (Exception,), {})("provider failed")
    provider = OpenAIProvider(
        client=FakeClient(FakeCompletions(error=sdk_error)),
        config=OpenAIConfig(api_key="fake"),
    )

    with pytest.raises(expected_type):
        await provider.generate(ChatRequest(messages=[Message.user("Hello")], model="example-model"))
