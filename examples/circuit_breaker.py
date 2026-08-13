"""Offline deterministic circuit breaker example."""

import asyncio

from modelcore.application import CircuitBreakerPolicy, CircuitBreakerProvider, CircuitState
from modelcore.exceptions import ProviderUnavailableError
from modelcore.models import ChatRequest, ChatResponse, Message


class Clock:
    value = 0.0

    def __call__(self) -> float:
        return self.value


class FakeProvider:
    def __init__(self) -> None:
        self.failures = 2

    async def generate(self, request: ChatRequest) -> ChatResponse:
        if self.failures:
            self.failures -= 1
            raise ProviderUnavailableError("offline provider unavailable")
        return ChatResponse("recovered", request.model, "fake", None)


async def main() -> None:
    clock = Clock()
    breaker = CircuitBreakerProvider(
        FakeProvider(),
        CircuitBreakerPolicy(failure_threshold=2, recovery_timeout=30),
        clock=clock,
    )
    request = ChatRequest([Message.user("offline")], model="test-model")
    for _ in range(2):
        try:
            await breaker.generate(request)
        except ProviderUnavailableError:
            pass
    print(breaker.state is CircuitState.OPEN)
    clock.value = 30
    response = await breaker.generate(request)
    print(response.content, breaker.state is CircuitState.CLOSED)


if __name__ == "__main__":
    asyncio.run(main())
