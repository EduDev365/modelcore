from modelcore.models import EmbeddingRequest


async def embed(provider):
    return await provider.embed(EmbeddingRequest(["hello"], model="your-embedding-model"))
