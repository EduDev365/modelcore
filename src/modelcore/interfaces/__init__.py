"""Contracts shared by ModelCore application and providers."""

from modelcore.interfaces.cache_backend import CacheBackend
from modelcore.interfaces.embedding_provider import EmbeddingProvider
from modelcore.interfaces.llm_provider import LLMProvider
from modelcore.interfaces.structured_output_provider import StructuredOutputProvider
from modelcore.interfaces.telemetry_sink import TelemetrySink
from modelcore.interfaces.tool_calling_provider import ToolCallingProvider

__all__ = [
    "CacheBackend",
    "EmbeddingProvider",
    "LLMProvider",
    "StructuredOutputProvider",
    "TelemetrySink",
    "ToolCallingProvider",
]
