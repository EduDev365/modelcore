from modelcore.config.openai import OpenAIConfig


def test_openai_config_does_not_include_api_key_in_repr() -> None:
    config = OpenAIConfig(api_key="not-a-real-key")

    assert "not-a-real-key" not in repr(config)
