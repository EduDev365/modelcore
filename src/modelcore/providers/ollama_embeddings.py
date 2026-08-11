from typing import Any

from modelcore.config.ollama import OllamaConfig
from modelcore.exceptions.provider import ProviderError, ProviderUnavailableError
from modelcore.models.embedding_request import EmbeddingRequest
from modelcore.models.embedding_response import EmbeddingResponse
from modelcore.models.embedding_usage import EmbeddingUsage
from modelcore.providers._ollama import (
    OLLAMA_RESPONSE_ERRORS,
    OLLAMA_TRANSPORT_ERRORS,
    create_ollama_client,
    map_ollama_response_error,
)


class OllamaEmbeddingProvider:
    """Adapter that translates between ModelCore and Ollama embeddings."""

    def __init__(self, config: OllamaConfig, client: Any | None = None) -> None:
        self._client = client if client is not None else create_ollama_client(config)

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        try:
            raw_response = await self._client.embed(
                model=request.model,
                input=list(request.texts),
            )
        except OLLAMA_RESPONSE_ERRORS as error:
            raise map_ollama_response_error(error) from error
        except OLLAMA_TRANSPORT_ERRORS as error:
            raise ProviderUnavailableError("Ollama provider unavailable") from error

        return self._normalize_response(raw_response, expected_count=len(request.texts))

    @staticmethod
    def _normalize_response(raw_response: Any, expected_count: int) -> EmbeddingResponse:
        try:
            embeddings = raw_response.embeddings
            if len(embeddings) != expected_count:
                raise ValueError("Embedding count does not match request inputs")
            input_tokens = getattr(raw_response, "prompt_eval_count", None)
            usage = EmbeddingUsage(input_tokens=input_tokens) if input_tokens is not None else None
            return EmbeddingResponse(
                embeddings=embeddings,
                model=raw_response.model,
                provider="ollama",
                usage=usage,
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise ProviderError("Ollama returned an invalid embedding response") from error
