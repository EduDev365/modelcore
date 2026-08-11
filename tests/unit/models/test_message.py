import pytest

from modelcore.models.message import Message


def test_message_accepts_supported_role_and_content() -> None:
    message = Message(role="user", content="Hello")

    assert message.role == "user"
    assert message.content == "Hello"


@pytest.mark.parametrize(
    "factory, role", [(Message.system, "system"), (Message.user, "user"), (Message.assistant, "assistant")]
)
def test_message_factories_create_messages(factory, role: str) -> None:
    message = factory("Hello")

    assert message == Message(role=role, content="Hello")


def test_message_rejects_unknown_role() -> None:
    with pytest.raises(ValueError, match="Unsupported message role"):
        Message(role="developer", content="Hello")
