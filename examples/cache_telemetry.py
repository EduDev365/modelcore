"""Compose offline cache telemetry around MemoryCache."""

import asyncio

from modelcore.application import MemoryCache, ObservableCacheBackend
from modelcore.models import CacheTelemetryEvent, ChatResponse


class PrintCacheTelemetrySink:
    async def emit(self, event: CacheTelemetryEvent) -> None:
        # CacheTelemetryEvent contains only safe operational metadata.
        print(event)


async def main() -> None:
    cache = ObservableCacheBackend(
        MemoryCache(),
        backend_name="memory",
        sink=PrintCacheTelemetrySink(),
    )
    response = ChatResponse(
        content="Hello",
        model="example",
        provider="example",
        usage=None,
        finish_reason="stop",
    )

    await cache.get("application-owned-key")  # miss
    await cache.set("application-owned-key", response, ttl=60)  # success
    await cache.get("application-owned-key")  # hit


if __name__ == "__main__":
    asyncio.run(main())
