"""Exceptions raised by ModelCore."""

from modelcore.exceptions.provider import (
    AuthenticationError,
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
