import json
from collections.abc import AsyncIterator, Sequence
from typing import Any

from modelcore.config.openai import OpenAIConfig
from modelcore.exceptions.provider import ProviderError
from modelcore.exceptions.tool import ToolValidationError
from modelcore.models.chat_request import ChatRequest
from modelcore.models.chat_response import ChatResponse
from modelcore.models.chat_stream_chunk import ChatStreamChunk
from modelcore.models.tools import ToolCall, ToolCallingResponse, ToolDefinition, ToolResult
from modelcore.models.usage import Usage
from modelcore.providers._openai import create_openai_client, raise_mapped_openai_error


class OpenAIProvider:
    """Adapter that translates between ModelCore and OpenAI chat completions."""

    def __init__(self, config: OpenAIConfig, client: Any | None = None) -> None:
        self._client = client if client is not None else create_openai_client(config)

    async def generate(self, request: ChatRequest) -> ChatResponse:
        try:
            raw_response = await self._client.chat.completions.create(
                model=request.model,
                messages=self._serialize_messages(request),
                temperature=request.temperature,
            )
        except Exception as error:
            raise_mapped_openai_error(error)

        return self._normalize_response(raw_response)

    async def generate_structured(
        self,
        request: ChatRequest,
        schema: dict[str, Any],
    ) -> ChatResponse:
        try:
            raw_response = await self._client.chat.completions.create(
                model=request.model,
                messages=self._serialize_messages(request),
                temperature=request.temperature,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "structured_output",
                        "schema": schema,
                        "strict": True,
                    },
                },
            )
        except Exception as error:
            raise_mapped_openai_error(error)

        return self._normalize_response(raw_response)

    async def generate_with_tools(self, request: ChatRequest, tools: Sequence[ToolDefinition]) -> ToolCallingResponse:
        try:
            raw = await self._client.chat.completions.create(
                model=request.model,
                messages=self._serialize_messages(request),
                temperature=request.temperature,
                tools=_serialize_tools(tools),
            )
        except Exception as error:
            raise_mapped_openai_error(error)
        return _normalize_tool_response(raw)

    async def continue_with_tool_results(
        self,
        request: ChatRequest,
        initial: ToolCallingResponse,
        results: Sequence[ToolResult],
        tools: Sequence[ToolDefinition],
    ) -> ToolCallingResponse:
        messages = (
            self._serialize_messages(request)
            + [
                {
                    "role": "assistant",
                    "content": initial.content,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(call.arguments, separators=(",", ":")),
                            },
                        }
                        for call in initial.tool_calls
                    ],
                }
            ]
            + [{"role": "tool", "tool_call_id": result.tool_call_id, "content": result.content} for result in results]
        )
        try:
            raw = await self._client.chat.completions.create(
                model=request.model, messages=messages, temperature=request.temperature, tools=_serialize_tools(tools)
            )
        except Exception as error:
            raise_mapped_openai_error(error)
        return _normalize_tool_response(raw)

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        try:
            raw_stream = await self._client.chat.completions.create(
                model=request.model,
                messages=self._serialize_messages(request),
                temperature=request.temperature,
                stream=True,
                stream_options={"include_usage": True},
            )
            async for raw_chunk in raw_stream:
                chunk = self._normalize_stream_chunk(raw_chunk)
                if chunk is not None:
                    yield chunk
        except Exception as error:
            raise_mapped_openai_error(error)

    @staticmethod
    def _serialize_messages(request: ChatRequest) -> list[dict[str, str]]:
        return [{"role": message.role, "content": message.content} for message in request.messages]

    @staticmethod
    def _normalize_response(raw_response: Any) -> ChatResponse:
        try:
            choice = raw_response.choices[0]
            raw_usage = raw_response.usage
            usage = OpenAIProvider._normalize_usage(raw_usage)
            return ChatResponse(
                content=choice.message.content or "",
                model=raw_response.model,
                provider="openai",
                usage=usage,
                finish_reason=choice.finish_reason,
            )
        except (AttributeError, IndexError, TypeError, ValueError) as error:
            raise ProviderError("OpenAI returned an invalid chat response") from error

    @staticmethod
    def _normalize_stream_chunk(raw_chunk: Any) -> ChatStreamChunk | None:
        try:
            choice = raw_chunk.choices[0] if raw_chunk.choices else None
            content_delta = choice.delta.content if choice is not None else ""
            finish_reason = choice.finish_reason if choice is not None else None
            raw_usage = getattr(raw_chunk, "usage", None)
            usage = OpenAIProvider._normalize_usage(raw_usage) if raw_usage is not None else None
            if content_delta is None:
                content_delta = ""
            if not content_delta and finish_reason is None and usage is None:
                return None
            return ChatStreamChunk(
                content_delta=content_delta,
                model=raw_chunk.model,
                provider="openai",
                finish_reason=finish_reason,
                usage=usage,
            )
        except (AttributeError, IndexError, TypeError, ValueError) as error:
            raise ProviderError("OpenAI returned an invalid chat stream chunk") from error

    @staticmethod
    def _normalize_usage(raw_usage: Any) -> Usage:
        return Usage(
            input_tokens=raw_usage.prompt_tokens,
            output_tokens=raw_usage.completion_tokens,
            total_tokens=raw_usage.total_tokens,
        )


def _serialize_tools(tools: Sequence[ToolDefinition]) -> list[dict[str, object]]:
    return [
        {
            "type": "function",
            "function": {"name": tool.name, "description": tool.description, "parameters": tool.json_schema},
        }
        for tool in tools
    ]


def _normalize_tool_response(raw: Any) -> ToolCallingResponse:
    try:
        choice = raw.choices[0]
        calls = []
        for raw_call in choice.message.tool_calls or []:
            arguments = json.loads(raw_call.function.arguments)
            if not isinstance(arguments, dict):
                raise ToolValidationError("Tool call arguments must be a JSON object")
            calls.append(ToolCall(raw_call.id, raw_call.function.name, arguments))
        return ToolCallingResponse(
            choice.message.content,
            tuple(calls),
            raw.model,
            "openai",
            OpenAIProvider._normalize_usage(raw.usage),
            choice.finish_reason,
        )
    except ToolValidationError:
        raise
    except (AttributeError, IndexError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ToolValidationError("OpenAI returned invalid tool call arguments") from error
