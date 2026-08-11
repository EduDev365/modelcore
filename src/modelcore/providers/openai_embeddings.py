from typing import Any

from modelcore.config.openai import OpenAIConfig
from modelcore.exceptions.provider import ProviderError
from modelcore.models.embedding_request import EmbeddingRequest
from modelcore.models.embedding_response import EmbeddingResponse
from modelcore.models.embedding_usage import EmbeddingUsage
from modelcore.providers._openai import create_openai_client, raise_mapped_openai_error


class OpenAIEmbeddingProvider:
    """Adapter that translates between ModelCore and OpenAI embeddings."""

    def __init__(self, config: OpenAIConfig, client: Any | None = None) -> None:
        self._client = client if client is not None else create_openai_client(config)

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        try:
            raw_response = await self._client.embeddings.create(
                model=request.model,
                input=list(request.texts),
            )
        except Exception as error:
            raise_mapped_openai_error(error)

        return self._normalize_response(raw_response, expected_count=len(request.texts))

    @staticmethod
    def _normalize_response(raw_response: Any, expected_count: int) -> EmbeddingResponse:
        try:
            items = sorted(raw_response.data, key=lambda item: item.index)
            if [item.index for item in items] != list(range(expected_count)):
                raise ValueError("Embedding indexes do not match request inputs")
            raw_usage = getattr(raw_response, "usage", None)
            usage = (
                EmbeddingUsage(
                    input_tokens=raw_usage.prompt_tokens,
                    total_tokens=getattr(raw_usage, "total_tokens", None),
                )
                if raw_usage is not None
                else None
            )
            return EmbeddingResponse(
                embeddings=[item.embedding for item in items],
                model=raw_response.model,
                provider="openai",
                usage=usage,
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise ProviderError("OpenAI returned an invalid embedding response") from error
