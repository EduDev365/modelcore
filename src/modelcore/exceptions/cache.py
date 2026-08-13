from modelcore.exceptions.provider import ModelCoreError


class CacheBackendError(ModelCoreError):
    """A cache backend failed while processing an operation."""


class CacheUnavailableError(CacheBackendError):
    """A cache backend could not be reached or timed out."""
