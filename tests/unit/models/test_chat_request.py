import pytest

from modelcore.models.chat_request import ChatRequest
from modelcore.models.message import Message


def test_chat_request_accepts_messages_and_model() -> None:
    request = ChatRequest(
        messages=[Message.user("Explain decorators.")],
        model="example-model",
    )

    assert request.messages == (Message.user("Explain decorators."),)
    assert request.model == "example-model"


def test_chat_request_rejects_empty_messages() -> None:
    with pytest.raises(ValueError, match="at least one message"):
        ChatRequest(messages=[], model="example-model")


def test_chat_request_rejects_blank_model() -> None:
    with pytest.raises(ValueError, match="model cannot be blank"):
        ChatRequest(messages=[Message.user("Hello")], model=" ")


@pytest.mark.parametrize("temperature", [-0.1, 2.1])
def test_chat_request_keeps_temperature_generic_for_provider_validation(temperature: float) -> None:
    request = ChatRequest(messages=[Message.user("Hello")], model="example-model", temperature=temperature)

    assert request.temperature == temperature
