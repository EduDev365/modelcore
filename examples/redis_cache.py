from redis.asyncio import Redis

from modelcore.application import CachingProvider, ResilientProvider, RetryPolicy
from modelcore.cache import RedisCache
from modelcore.interfaces import LLMProvider


def build_cached_provider(provider: LLMProvider, client: Redis) -> CachingProvider:
    """Compose a shared Redis cache without opening a connection during import."""
    reliable = ResilientProvider(provider, RetryPolicy())
    cache = RedisCache(client, namespace="myapp:modelcore:")
    return CachingProvider(reliable, cache, provider_key="provider:model", ttl=60)
