"""Emit ModelCore generation spans through an application-managed tracer.

Install ``modelcore[otel]``. Configure an SDK and exporter in the application
if spans should be exported; this example requires neither a collector nor one.
"""

import os
import sys
from pathlib import Path

# Running this file directly places ``examples/`` first on sys.path, where this
# file would otherwise shadow the installed ``opentelemetry`` package.
sys.path.remove(str(Path(__file__).parent))

from opentelemetry import trace

from modelcore.application import TelemetryProvider
from modelcore.config import OpenAIConfig
from modelcore.models import ChatRequest, Message
from modelcore.providers import OpenAIProvider
from modelcore.telemetry.opentelemetry import OpenTelemetrySink


async def main() -> None:
    # The application configures its tracer provider and exporters. ModelCore does
    # not set global OpenTelemetry state or require a collector.
    tracer = trace.get_tracer("my_application.modelcore")
    provider = OpenAIProvider(OpenAIConfig(api_key=os.environ["OPENAI_API_KEY"]))
    observed = TelemetryProvider(provider, OpenTelemetrySink(tracer))

    response = await observed.generate(ChatRequest([Message.user("Hello")], model="gpt-5"))
    print(response.content)
