from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from modelcore.config.ollama import OllamaConfig
from modelcore.config.openai import OpenAIConfig
from modelcore.models import ChatRequest, Message, ToolDefinition
from modelcore.providers.ollama import OllamaProvider
from modelcore.providers.openai import OpenAIProvider


class Args(BaseModel):
    city: str


async def weather(city: str) -> str:
    return city


TOOL = ToolDefinition("weather", "Get weather", Args, weather)


class OpenAICompletions:
    def __init__(self, response):
        self.response, self.calls = response, []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class OpenAIClient:
    def __init__(self, response):
        self.chat = SimpleNamespace(completions=OpenAICompletions(response))


class OllamaClient:
    def __init__(self, response):
        self.response, self.calls = response, []

    async def chat(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def request():
    return ChatRequest([Message.user("weather")], model="test", temperature=0.2)


@pytest.mark.asyncio
async def test_openai_normalizes_multiple_calls_and_sends_function_schema() -> None:
    raw = SimpleNamespace(
        model="test",
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2, total_tokens=5),
        choices=[
            SimpleNamespace(
                finish_reason="tool_calls",
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(id="a", function=SimpleNamespace(name="weather", arguments='{"city":"A"}')),
                        SimpleNamespace(id="b", function=SimpleNamespace(name="weather", arguments='{"city":"B"}')),
                    ],
                ),
            )
        ],
    )
    client = OpenAIClient(raw)
    provider = OpenAIProvider(OpenAIConfig(api_key="x"), client=client)
    result = await provider.generate_with_tools(request(), [TOOL])
    assert [call.arguments for call in result.tool_calls] == [{"city": "A"}, {"city": "B"}]
    assert client.chat.completions.calls[0]["tools"] == [
        {
            "type": "function",
            "function": {"name": "weather", "description": "Get weather", "parameters": Args.model_json_schema()},
        }
    ]


@pytest.mark.asyncio
async def test_ollama_normalizes_calls_and_sends_schema() -> None:
    raw = SimpleNamespace(
        model="test",
        prompt_eval_count=3,
        eval_count=2,
        done_reason="tool_calls",
        message=SimpleNamespace(
            content="", tool_calls=[SimpleNamespace(function=SimpleNamespace(name="weather", arguments={"city": "A"}))]
        ),
    )
    client = OllamaClient(raw)
    provider = OllamaProvider(OllamaConfig(), client=client)
    result = await provider.generate_with_tools(request(), [TOOL])
    assert result.tool_calls[0].id is None
    assert result.tool_calls[0].arguments == {"city": "A"}
    assert client.calls[0]["tools"] == [
        {
            "type": "function",
            "function": {"name": "weather", "description": "Get weather", "parameters": Args.model_json_schema()},
        }
    ]
