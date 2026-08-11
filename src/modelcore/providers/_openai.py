from typing import Any

from modelcore.config.openai import OpenAIConfig
from modelcore.exceptions.provider import (
    AuthenticationError,
    ProviderError,
    ProviderUnavailableError,
    RateLimitError,
)


def create_openai_client(config: OpenAIConfig) -> Any:
    try:
        from openai import AsyncOpenAI
    except ModuleNotFoundError as error:
        raise ProviderError("The openai package is required for OpenAI providers") from error

    client_options: dict[str, Any] = {"api_key": config.api_key}
    if config.timeout is not None:
        client_options["timeout"] = config.timeout
    return AsyncOpenAI(**client_options)


def map_openai_error(error: Exception) -> Exception:
    error_name = type(error).__name__
    if error_name == "AuthenticationError":
        return AuthenticationError("OpenAI authentication failed")
    if error_name == "RateLimitError":
        return RateLimitError("OpenAI rate limit exceeded")
    if error_name in {"APIError", "APIConnectionError", "InternalServerError"}:
        return ProviderUnavailableError("OpenAI provider unavailable")
    return error


def raise_mapped_openai_error(error: Exception) -> None:
    mapped_error = map_openai_error(error)
    if mapped_error is error:
        raise error
    raise mapped_error from error
