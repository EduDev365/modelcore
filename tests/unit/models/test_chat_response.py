import pytest

from modelcore.models.chat_response import ChatResponse
from modelcore.models.usage import Usage


def test_chat_response_contains_normalized_generation_data() -> None:
    usage = Usage(input_tokens=12, output_tokens=5)
    response = ChatResponse(
        content="Generated answer",
        model="example-model",
        provider="example-provider",
        usage=usage,
        finish_reason="stop",
    )

    assert response.content == "Generated answer"
    assert response.model == "example-model"
    assert response.provider == "example-provider"
    assert response.usage is usage
    assert response.finish_reason == "stop"


def test_chat_response_finish_reason_is_optional() -> None:
    response = ChatResponse(
        content="Generated answer",
        model="example-model",
        provider="example-provider",
        usage=Usage(input_tokens=1, output_tokens=1),
    )

    assert response.finish_reason is None


def test_chat_response_allows_missing_usage_when_a_provider_does_not_report_it() -> None:
    response = ChatResponse(
        content="Generated answer",
        model="example-model",
        provider="example-provider",
        usage=None,
    )

    assert response.usage is None


def test_chat_response_requires_internal_usage() -> None:
    with pytest.raises(TypeError, match="Usage"):
        ChatResponse(
            content="Generated answer",
            model="example-model",
            provider="example-provider",
            usage="invalid",  # type: ignore[arg-type]
        )
