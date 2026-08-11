import pytest

from modelcore.models.chat_stream_chunk import ChatStreamChunk
from modelcore.models.usage import Usage


def test_chat_stream_chunk_represents_an_incremental_delta() -> None:
    chunk = ChatStreamChunk(
        content_delta="Hello",
        model="example-model",
        provider="example-provider",
    )

    assert chunk.content_delta == "Hello"
    assert chunk.model == "example-model"
    assert chunk.provider == "example-provider"
    assert chunk.finish_reason is None
    assert chunk.usage is None


def test_chat_stream_chunk_keeps_final_metadata_without_content() -> None:
    usage = Usage(input_tokens=3, output_tokens=2)
    chunk = ChatStreamChunk(
        content_delta="",
        model="example-model",
        provider="example-provider",
        finish_reason="stop",
        usage=usage,
    )

    assert chunk.finish_reason == "stop"
    assert chunk.usage is usage


def test_chat_stream_chunk_rejects_non_usage_metadata() -> None:
    with pytest.raises(TypeError, match="Usage"):
        ChatStreamChunk(
            content_delta="Hello",
            model="example-model",
            provider="example-provider",
            usage="invalid",  # type: ignore[arg-type]
        )
