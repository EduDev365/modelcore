import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]


@pytest.mark.parametrize(
    "example",
    [
        "cache_telemetry.py",
        "circuit_breaker.py",
        "resilience_telemetry.py",
        "routing_telemetry.py",
    ],
)
def test_offline_example_runs_without_network(example: str) -> None:
    subprocess.run([sys.executable, str(ROOT / "examples" / example)], check=True, cwd=ROOT)


def test_every_example_compiles() -> None:
    for example in (ROOT / "examples").glob("*.py"):
        compile(example.read_bytes(), str(example), "exec")
