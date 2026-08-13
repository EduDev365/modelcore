"""Offline routing telemetry example using a fake provider."""

import asyncio

from modelcore.application import CheapPolicy, ModelCandidate, RoutingProvider
from modelcore.models import ChatRequest, ChatResponse, Message


class FakeProvider:
    async def generate(self, request: ChatRequest) -> ChatResponse:
        return ChatResponse(content="offline response", model=request.model, provider="fake", usage=None)


class PrintingRoutingSink:
    async def emit(self, event) -> None:
        print(f"selected={event.candidate} model={event.model} policy={event.policy}")


async def main() -> None:
    provider = FakeProvider()
    router = RoutingProvider(
        CheapPolicy(),
        [
            ModelCandidate("openai-cheap", provider, "gpt-test", 1, 2, 3),
            ModelCandidate("local-quality", provider, "local-test", 2, 4, 5),
        ],
        telemetry_sink=PrintingRoutingSink(),
    )
    await router.generate(ChatRequest([Message.user("offline")], model="requested-model"))


if __name__ == "__main__":
    asyncio.run(main())
