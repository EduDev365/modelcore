import asyncio
import os
import uuid

import pytest

from modelcore.cache import RedisCache
from modelcore.models import ChatResponse, Usage

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_redis_cache_round_trip_and_ttl() -> None:
    if os.getenv("MODELCORE_RUN_REDIS_INTEGRATION") != "1":
        pytest.skip("Set MODELCORE_RUN_REDIS_INTEGRATION=1 to run")

    redis = pytest.importorskip("redis.asyncio")
    client = redis.Redis.from_url(os.getenv("MODELCORE_REDIS_URL", "redis://localhost:6379/0"))
    namespace = f"modelcore:integration:{uuid.uuid4()}:"
    cache = RedisCache(client, namespace=namespace)
    response = ChatResponse(
        content="integration",
        model="redis-test",
        provider="integration",
        usage=Usage(input_tokens=1, output_tokens=1),
        finish_reason="stop",
    )
    try:
        await cache.set("key", response, ttl=0.1)
        assert await cache.get("key") == response
        await asyncio.sleep(0.2)
        assert await cache.get("key") is None
    finally:
        await client.delete(f"{namespace}key")
        await client.aclose()
