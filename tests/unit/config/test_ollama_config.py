import pytest

from modelcore.config.ollama import OllamaConfig


def test_ollama_config_uses_the_local_default_host() -> None:
    assert OllamaConfig().base_url == "http://localhost:11434"


def test_ollama_config_rejects_a_blank_base_url() -> None:
    with pytest.raises(ValueError, match="base_url cannot be blank"):
        OllamaConfig(base_url=" ")
