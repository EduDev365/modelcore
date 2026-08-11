from modelcore.models import ChatRequest, Message


async def stream(provider) -> None:
    request = ChatRequest([Message.user("Hello")], model="your-model")
    async for chunk in provider.stream(request):
        print(chunk.content_delta, end="")
