import pytest

from modelcore.models.usage import Usage


def test_usage_calculates_total_when_it_is_omitted() -> None:
    usage = Usage(input_tokens=10, output_tokens=4)

    assert usage.input_tokens == 10
    assert usage.output_tokens == 4
    assert usage.total_tokens == 14


def test_usage_accepts_a_consistent_explicit_total() -> None:
    assert Usage(input_tokens=10, output_tokens=4, total_tokens=14).total_tokens == 14


def test_usage_rejects_an_inconsistent_total() -> None:
    with pytest.raises(ValueError, match="total_tokens must equal"):
        Usage(input_tokens=10, output_tokens=4, total_tokens=99)


@pytest.mark.parametrize("field", ["input_tokens", "output_tokens", "total_tokens"])
def test_usage_rejects_negative_token_counts(field: str) -> None:
    values = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
    values[field] = -1

    with pytest.raises(ValueError, match="cannot be negative"):
        Usage(**values)
