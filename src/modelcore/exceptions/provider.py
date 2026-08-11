class ModelCoreError(Exception):
    """Base class for errors intentionally exposed by ModelCore."""


class ProviderError(ModelCoreError):
    """A provider failed while processing a request."""


class AuthenticationError(ProviderError):
    """A provider rejected the configured credentials."""


class RateLimitError(ProviderError):
    """A provider rejected the request because of rate limiting."""


class ProviderUnavailableError(ProviderError):
    """A provider could not process a request because it is temporarily unavailable."""


class GenerationTimeoutError(ProviderError):
    """A generation attempt exceeded its configured timeout."""


class StructuredOutputError(ModelCoreError):
    """Structured output could not be parsed or validated against its schema."""
