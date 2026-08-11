from typing import get_type_hints

from modelcore.interfaces.telemetry_sink import TelemetrySink
from modelcore.models.telemetry import GenerationTelemetryEvent


def test_telemetry_sink_emits_generation_events() -> None:
    hints = get_type_hints(TelemetrySink.emit)

    assert hints["event"] is GenerationTelemetryEvent
    assert "return" in hints
