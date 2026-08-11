import os

import pytest

from modelcore.config import OllamaConfig, OpenAIConfig
from modelcore.models import ChatRequest, ChatResponse, Message
from modelcore.providers import OllamaProvider, OpenAIProvider

pytestmark = pytest.mark.integration


def _enabled() -> bool:
    return os.getenv("MODELCORE_RUN_INTEGRATION") == "1"


@pytest.mark.asyncio
async def test_openai_chat_integration() -> None:
    if not _enabled() or not os.getenv("OPENAI_API_KEY"):
        pytest.skip("Set MODELCORE_RUN_INTEGRATION=1 and OPENAI_API_KEY to run")
    provider = OpenAIProvider(OpenAIConfig(api_key=os.environ["OPENAI_API_KEY"]))
    response = await provider.generate(
        ChatRequest([Message.user("Reply with OK")], model=os.getenv("OPENAI_MODEL", "gpt-5"))
    )
    assert isinstance(response, ChatResponse)


@pytest.mark.asyncio
async def test_ollama_chat_integration() -> None:
    if not _enabled() or not os.getenv("OLLAMA_MODEL"):
        pytest.skip("Set MODELCORE_RUN_INTEGRATION=1 and OLLAMA_MODEL to run")
    provider = OllamaProvider(OllamaConfig(base_url=os.getenv("OLLAMA_HOST", "http://localhost:11434")))
    response = await provider.generate(ChatRequest([Message.user("Reply with OK")], model=os.environ["OLLAMA_MODEL"]))
    assert isinstance(response, ChatResponse)
