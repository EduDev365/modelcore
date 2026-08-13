# Changelog

## Unreleased

- Added safe routing decision telemetry and an optional OpenTelemetry routing sink.
- Added a generate-only circuit breaker with deterministic recovery and concurrent half-open protection.
- Hardened retry, circuit breaker, fallback, cache, telemetry, and routing composition semantics.
- Expanded public documentation, packaging checks, and cross-platform CI coverage for release readiness.

## 1.4.0

- Added optional `OpenTelemetrySink` support through the `otel` extra.
- Generation telemetry can emit safe operational spans without prompts, generated content, or secrets.
- No known breaking changes.

## 1.0.0

- Provider-agnostic chat, streaming, embeddings, structured output, resilience, memory cache, fallback, telemetry, routing, and safe tool calling.
