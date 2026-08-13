"""Exceptions raised by ModelCore."""

from modelcore.exceptions.cache import CacheBackendError, CacheUnavailableError
from modelcore.exceptions.provider import (
    AuthenticationError,
    CircuitOpenError,
    GenerationTimeoutError,
    ModelCoreError,
    ProviderError,
    ProviderUnavailableError,
    RateLimitError,
    StructuredOutputError,
)
from modelcore.exceptions.tool import (
    ToolError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolRoundLimitError,
    ToolValidationError,
)

__all__ = [
    "AuthenticationError",
    "CacheBackendError",
    "CacheUnavailableError",
    "CircuitOpenError",
    "GenerationTimeoutError",
    "ModelCoreError",
    "ProviderError",
    "ProviderUnavailableError",
    "RateLimitError",
    "StructuredOutputError",
    "ToolError",
    "ToolExecutionError",
    "ToolNotFoundError",
    "ToolRoundLimitError",
    "ToolValidationError",
]
