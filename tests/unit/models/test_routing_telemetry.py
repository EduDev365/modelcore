from dataclasses import FrozenInstanceError

import pytest

from modelcore.models import RoutingTelemetryEvent


def test_routing_event_is_immutable_and_contains_only_decision_metadata() -> None:
    event = RoutingTelemetryEvent("balanced", "candidate-a", "model-a", 1, 2, 1.5, 1.0, 2.0, 3.0)

    with pytest.raises(FrozenInstanceError):
        event.model = "prompt"  # type: ignore[misc]

    assert "Hello" not in repr(event)
    assert "response" not in repr(event)
