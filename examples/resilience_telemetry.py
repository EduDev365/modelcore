"""Observe retry and fallback composition without network access."""

import asyncio
from collections.abc import AsyncIterator

from modelcore.application import FallbackProvider, ResilientProvider, RetryPolicy
from modelcore.exceptions import ProviderUnavailableError
from modelcore.models import (
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    FallbackTelemetryEvent,
    Message,
    RetryTelemetryEvent,
)


class PrintRetrySink:
    async def emit(self, event: RetryTelemetryEvent) -> None:
        print("retry:", event)


class PrintFallbackSink:
    async def emit(self, event: FallbackTelemetryEvent) -> None:
        print("fallback:", event)


class FailingProvider:
    async def generate(self, request: ChatRequest) -> ChatResponse:
        raise ProviderUnavailableError("offline example failure")

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        if False:
            yield ChatStreamChunk(content_delta="", model=request.model, provider="primary")


class SuccessfulProvider:
    async def generate(self, request: ChatRequest) -> ChatResponse:
        return ChatResponse("Hello", request.model, "secondary", None, "stop")

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        yield ChatStreamChunk(content="Hello", model=request.model, provider="secondary")


async def main() -> None:
    retry_sink = PrintRetrySink()
    providers = [
        ResilientProvider(
            FailingProvider(),
            RetryPolicy(max_attempts=2, base_delay=0),
            provider_name="primary",
            telemetry_sink=retry_sink,
        ),
        ResilientProvider(
            SuccessfulProvider(),
            RetryPolicy(max_attempts=2, base_delay=0),
            provider_name="secondary",
            telemetry_sink=retry_sink,
        ),
    ]
    provider = FallbackProvider(
        providers,
        provider_names=["primary", "secondary"],
        telemetry_sink=PrintFallbackSink(),
    )
    response = await provider.generate(ChatRequest([Message.user("Hello")], model="example"))
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
