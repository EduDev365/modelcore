from pydantic import BaseModel

from modelcore.application import ToolExecutor, ToolGeneration, ToolRegistry
from modelcore.models import ChatRequest, Message, ToolDefinition


class WeatherArgs(BaseModel):
    city: str


async def weather(city: str) -> str:
    return f"Weather for {city}"


async def generate(provider):
    tool = ToolDefinition("weather", "Get weather", WeatherArgs, weather)
    return await ToolGeneration(provider, ToolExecutor(ToolRegistry([tool]))).generate(
        ChatRequest([Message.user("Weather in Curitiba")], model="your-model"), [tool]
    )
