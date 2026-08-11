from pydantic import BaseModel

from modelcore.application import StructuredGeneration
from modelcore.models import ChatRequest, Message


class Answer(BaseModel):
    text: str


async def generate(provider) -> Answer:
    return await StructuredGeneration(provider).generate(
        ChatRequest([Message.user("Answer briefly")], model="your-model"), Answer
    )
