from typing import Any

from modelcore.config.ollama import OllamaConfig
from modelcore.exceptions.provider import ProviderError, ProviderUnavailableError

try:
    from httpx import HTTPError
    from ollama import ResponseError
except ModuleNotFoundError:
    OLLAMA_RESPONSE_ERRORS: tuple[type[Exception], ...] = ()
    OLLAMA_TRANSPORT_ERRORS: tuple[type[Exception], ...] = ()
else:
    OLLAMA_RESPONSE_ERRORS = (ResponseError,)
    OLLAMA_TRANSPORT_ERRORS = (HTTPError,)


def create_ollama_client(config: OllamaConfig) -> Any:
    try:
        from ollama import AsyncClient
    except ModuleNotFoundError as error:
        raise ProviderError("The ollama package is required for Ollama providers") from error

    client_options: dict[str, Any] = {}
    if config.timeout is not None:
        client_options["timeout"] = config.timeout
    return AsyncClient(host=config.base_url, **client_options)


def map_ollama_response_error(error: Exception) -> ProviderError:
    if getattr(error, "status_code", 0) >= 500:
        return ProviderUnavailableError("Ollama provider unavailable")
    return ProviderError("Ollama request failed")
