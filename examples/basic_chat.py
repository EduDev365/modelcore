import os

from modelcore.config import OpenAIConfig
from modelcore.models import ChatRequest, Message
from modelcore.providers import OpenAIProvider


async def main() -> None:
    provider = OpenAIProvider(OpenAIConfig(api_key=os.environ["OPENAI_API_KEY"]))
    response = await provider.generate(ChatRequest([Message.user("Hello")], model=os.getenv("OPENAI_MODEL", "gpt-5")))
    print(response.content)
