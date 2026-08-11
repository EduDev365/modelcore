from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Usage:
    """Token usage normalized across model providers."""

    input_tokens: int
    output_tokens: int
    total_tokens: int | None = None

    def __post_init__(self) -> None:
        for field_name in ("input_tokens", "output_tokens", "total_tokens"):
            value = getattr(self, field_name)
            if value is not None and value < 0:
                raise ValueError(f"{field_name} cannot be negative")

        expected_total = self.input_tokens + self.output_tokens
        if self.total_tokens is None:
            object.__setattr__(self, "total_tokens", expected_total)
        elif self.total_tokens != expected_total:
            raise ValueError("total_tokens must equal input_tokens + output_tokens")
