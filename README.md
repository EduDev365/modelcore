# ModelCore

A lightweight, production-oriented model infrastructure library for Python AI applications.

ModelCore provides an async, provider-agnostic layer between applications and model SDKs. Applications use normalized
requests, responses, protocols, and explicit infrastructure wrappers while adapters isolate provider-specific APIs.
The result is reusable model infrastructure without turning the library into an agent or RAG framework.

```text
Application
    |
    v
ModelCore contracts and normalized models
    |
    +-- explicit wrappers: cache, telemetry, routing, fallback, circuit breaker, retry
    |
    v
Provider adapter
    +-- OpenAI
    +-- Ollama
```

Wrapper order is chosen by the application; the diagram does not prescribe a global pipeline.

## Why ModelCore?

Depending directly on a provider SDK spreads provider types, error handling, and operational policy throughout an
application. ModelCore keeps application code dependent on stable internal contracts:

```text
application -> ModelCore contracts -> provider adapter -> provider SDK
```

This makes provider replacement and infrastructure composition explicit while minimizing SDK leakage.

## Features

- Core: chat generation, streaming, embeddings, structured output, and bounded tool calling.
- Resilience: retry with exponential backoff, timeout, fallback, and circuit breaker.
- Infrastructure: memory and optional Redis cache, model routing, safe telemetry, and optional OpenTelemetry adapters.
- Providers: OpenAI and Ollama adapters, each installed through an optional extra.
- Quality: typed public APIs, `py.typed`, deterministic offline tests, and segregated protocols.

## Installation

Install the core contracts and wrappers:

```bash
pip install modelcore
```

Install only the integrations an application needs:

```bash
pip install "modelcore[openai]"
pip install "modelcore[ollama]"
pip install "modelcore[otel]"
pip install "modelcore[redis]"
```

## Quickstart

Use environment-managed credentials rather than embedding secrets in source code:

```python
import os

from modelcore.config import OpenAIConfig
from modelcore.models import ChatRequest, Message
from modelcore.providers import OpenAIProvider

provider = OpenAIProvider(OpenAIConfig(api_key=os.environ["OPENAI_API_KEY"]))
request = ChatRequest([Message.user("Explain dependency inversion briefly.")], model="gpt-5")
response = await provider.generate(request)
print(response.content)
```

Ollama follows the same normalized request/response model with `OllamaConfig` and `OllamaProvider`.

## Provider abstraction

The main contracts are intentionally segregated: `LLMProvider`, `EmbeddingProvider`, `StructuredOutputProvider`, and
`ToolCallingProvider`. Applications can implement these protocols with local fakes or additional adapters without
inheriting from framework base classes. OpenAI and Ollama SDK objects remain inside their adapters.

## Resilience and composition

Retry recovers an individual operation. A circuit breaker avoids repeatedly calling a persistently unavailable
provider. Fallback moves to the next provider after an eligible failure. A common arrangement is:

```python
from modelcore.application import (
    CircuitBreakerPolicy,
    CircuitBreakerProvider,
    FallbackProvider,
    ResilientProvider,
    RetryPolicy,
)

primary = CircuitBreakerProvider(
    ResilientProvider(openai, RetryPolicy()),
    CircuitBreakerPolicy(),
)
secondary = CircuitBreakerProvider(
    ResilientProvider(ollama, RetryPolicy()),
    CircuitBreakerPolicy(),
)
provider = FallbackProvider([primary, secondary])
```

With `CircuitBreakerProvider(ResilientProvider(provider, ...), ...)`, retry attempts happen first and the breaker sees
one final operation result. With the inverse order, `ResilientProvider(CircuitBreakerProvider(provider, ...), ...)`,
an open circuit fails immediately because `CircuitOpenError` is not retryable. It is fallback-eligible, allowing the
next configured provider to run. These arrangements are deliberately not equivalent.

`CancelledError` is preserved across wrappers. Authentication and generic provider errors remain non-transient.
Rate limits, provider unavailability, and generation timeouts are the transient categories used by retry and breaker.

## Cache

`CachingProvider` accepts the `CacheBackend` protocol, uses deterministic request keys, supports TTL, and
short-circuits every lower layer on a hit. `MemoryCache` is process-local. The optional `RedisCache` adapter accepts an
application-owned `redis.asyncio` client; Redis failures remain explicit rather than silently degrading.

```python
from modelcore.application import CachingProvider, MemoryCache

cached = CachingProvider(provider, MemoryCache(), provider_key="openai:gpt-5", ttl=60)
```

Cached responses contain generated content. Applications must choose storage access controls, retention, and
encryption appropriate to their data. Stampede protection is process-local; no distributed lock is provided.

## Routing

`RoutingProvider` selects one configured `ModelCandidate` before generation. Built-in policies are intentionally
simple and deterministic:

- `CheapPolicy`: lowest configured cost score.
- `FastPolicy`: lowest configured latency score.
- `QualityPolicy`: highest configured quality score.
- `BalancedPolicy`: highest quality minus cost minus latency score.

Ties preserve candidate order. Routing uses configured metadata only; it does not inspect provider health, live
pricing, or online latency.

Routing telemetry describes that initial decision using the selected candidate key, model, configured scores,
position, policy identity, and decision duration. It is separate from generation telemetry: a selected candidate may
later retry, fail, or fall back without rewriting the routing event.

## Telemetry and privacy

Generation, cache, retry, fallback, and routing telemetry use separate event and sink contracts. Existing sinks are
best-effort: sink failure does not change provider selection, responses, or propagated provider errors.

The optional `modelcore[otel]` adapters emit application-managed OpenTelemetry spans without configuring global SDK
state or requiring a collector. Standard events deliberately exclude prompts, messages, generated content, cache
keys and values, tool arguments, credentials, endpoints, provider representations, and raw exception messages.
Configured provider, model, candidate, and policy names are operational metadata controlled by the application.

## Tool calling

Tool calling is explicit and bounded. `ToolDefinition` describes an allowlisted tool and Pydantic argument model,
`ToolRegistry` resolves it, `ToolExecutor` validates and invokes it, and `ToolGeneration` performs at most one tool
execution round.

```python
from pydantic import BaseModel

from modelcore.application import ToolExecutor, ToolGeneration, ToolRegistry
from modelcore.models import ToolDefinition


class WeatherArgs(BaseModel):
    city: str


async def weather(city: str) -> str:
    return f"Weather for {city}"


tool = ToolDefinition("weather", "Get weather", WeatherArgs, weather)
tool_generation = ToolGeneration(provider, ToolExecutor(ToolRegistry([tool])))
result = await tool_generation.generate(request, [tool])
```

ModelCore does not execute arbitrary generated code and is not an agent framework.

## Streaming

OpenAI and Ollama adapters normalize incremental chunks through `ChatStreamChunk`. Cache, telemetry, retry, and
circuit-breaker wrappers delegate streaming according to their documented scope rather than buffering or replaying
partial streams. Circuit breaking currently applies to `generate()` only; `stream()` is delegated directly.

## Wrapper order matters

Composition remains under consumer control:

- Cache outside a composition can bypass routing, fallback, breaker, retry, and providers on a hit.
- Telemetry outside cache observes cache-served generations; telemetry inside cache is bypassed on a hit.
- Retry inside a breaker makes the breaker observe only the final result after retries.
- Fallback handles eligible failures, including an explicitly open circuit, without aggregating unrelated errors.

ModelCore does not provide a pipeline builder or impose one universal order.

## Design principles

- Async-first interfaces and implementations.
- Provider isolation and minimal SDK leakage.
- Interface segregation and normalized internal models.
- Composition over inheritance.
- Explicit operational identities and error semantics.
- Monotonic, injectable clocks for deterministic time-based behavior.
- Small fakes and offline deterministic tests.

## Security responsibilities

ModelCore handles model-infrastructure concerns: secret-safe configuration representations, controlled package
contents, predictable errors, timeouts, resilience, schema validation, and privacy-conscious telemetry defaults.

Applications remain responsible for authentication, authorization, tenant isolation, user rate limiting, quotas,
billing, anti-abuse controls, API gateway policy, network security, and the confidentiality of prompts, responses,
tool outputs, and cached content. Architecture or model names should not be treated as security boundaries.

## Scope and non-goals

ModelCore is not a RAG or document framework, vector database abstraction, agent framework, workflow engine,
authentication layer, API gateway, SaaS platform, or official provider SDK. It does not include health-aware routing,
distributed circuit breakers, background health probes, user quotas, or automatic wrapper composition.

## Examples

The [`examples/`](examples/) directory contains chat, streaming, embeddings, structured output, tool calling,
resilience, circuit breaker, fallback, cache telemetry, routing, routing telemetry, Redis, and OpenTelemetry examples.
Offline examples use fakes and synthetic data; provider examples read credentials from environment variables.

## Testing and development

The core suite uses deterministic unit and integration tests without real network or provider dependencies. Real
provider and Redis integration tests are opt-in through explicit environment variables.

```bash
python -m pip install -e ".[dev,test,openai,ollama,otel,redis]"
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m mypy src/modelcore
python -m compileall -q src/modelcore
python -m pip check
python -m build
```

CI covers Linux and Windows on Python 3.11 through 3.14. See [RELEASE.md](RELEASE.md) for the reviewed release process.

## License

ModelCore is available under the [MIT License](LICENSE).
