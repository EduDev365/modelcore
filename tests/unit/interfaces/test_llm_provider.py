from collections.abc import AsyncIterator
from typing import get_args, get_origin, get_type_hints

from modelcore.interfaces.llm_provider import LLMProvider
from modelcore.models.chat_stream_chunk import ChatStreamChunk


def test_llm_provider_declares_generation_and_streaming() -> None:
    annotations = get_type_hints(LLMProvider.generate)

    assert "request" in annotations
    assert "return" in annotations
    stream_return_type = get_type_hints(LLMProvider.stream)["return"]
    assert get_origin(stream_return_type) is AsyncIterator
    assert get_args(stream_return_type) == (ChatStreamChunk,)
    assert not hasattr(LLMProvider, "embed")
