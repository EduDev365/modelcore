from typing import get_type_hints

from modelcore.interfaces.structured_output_provider import StructuredOutputProvider


def test_structured_output_provider_declares_only_structured_generation() -> None:
    annotations = get_type_hints(StructuredOutputProvider.generate_structured)

    assert "request" in annotations
    assert "schema" in annotations
    assert "return" in annotations
    assert not hasattr(StructuredOutputProvider, "embed")
    assert not hasattr(StructuredOutputProvider, "stream")
