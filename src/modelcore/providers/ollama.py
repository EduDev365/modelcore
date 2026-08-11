from collections.abc import AsyncIterator, Sequence
from typing import Any

from modelcore.config.ollama import OllamaConfig
from modelcore.exceptions.provider import ProviderError, ProviderUnavailableError
from modelcore.models.chat_request import ChatRequest
from modelcore.models.chat_response import ChatResponse
from modelcore.models.chat_stream_chunk import ChatStreamChunk
from modelcore.models.tools import ToolCall, ToolCallingResponse, ToolDefinition, ToolResult
from modelcore.models.usage import Usage
from modelcore.providers._ollama import (
    OLLAMA_RESPONSE_ERRORS,
    OLLAMA_TRANSPORT_ERRORS,
    create_ollama_client,
    map_ollama_response_error,
)


class OllamaProvider:
    """Adapter that translates between ModelCore and Ollama chat responses."""

    def __init__(self, config: OllamaConfig, client: Any | None = None) -> None:
        self._client = client if client is not None else create_ollama_client(config)

    async def generate(self, request: ChatRequest) -> ChatResponse:
        try:
            raw_response = await self._client.chat(
                model=request.model,
                messages=self._serialize_messages(request),
                options={"temperature": request.temperature},
                stream=False,
            )
        except OLLAMA_RESPONSE_ERRORS as error:
            raise map_ollama_response_error(error) from error
        except OLLAMA_TRANSPORT_ERRORS as error:
            raise ProviderUnavailableError("Ollama provider unavailable") from error

        return self._normalize_response(raw_response)

    async def generate_structured(
        self,
        request: ChatRequest,
        schema: dict[str, Any],
    ) -> ChatResponse:
        try:
            raw_response = await self._client.chat(
                model=request.model,
                messages=self._serialize_messages(request),
                options={"temperature": request.temperature},
                format=schema,
                stream=False,
            )
        except OLLAMA_RESPONSE_ERRORS as error:
            raise map_ollama_response_error(error) from error
        except OLLAMA_TRANSPORT_ERRORS as error:
            raise ProviderUnavailableError("Ollama provider unavailable") from error

        return self._normalize_response(raw_response)

    async def generate_with_tools(self, request: ChatRequest, tools: Sequence[ToolDefinition]) -> ToolCallingResponse:
        try:
            raw = await self._client.chat(
                model=request.model,
                messages=self._serialize_messages(request),
                options={"temperature": request.temperature},
                tools=_serialize_tools(tools),
                stream=False,
            )
        except OLLAMA_RESPONSE_ERRORS as error:
            raise map_ollama_response_error(error) from error
        except OLLAMA_TRANSPORT_ERRORS as error:
            raise ProviderUnavailableError("Ollama provider unavailable") from error
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
                    "content": initial.content or "",
                    "tool_calls": [
                        {"function": {"name": call.name, "arguments": call.arguments}} for call in initial.tool_calls
                    ],
                }
            ]
            + [{"role": "tool", "tool_name": result.name, "content": result.content} for result in results]
        )
        try:
            raw = await self._client.chat(
                model=request.model,
                messages=messages,
                options={"temperature": request.temperature},
                tools=_serialize_tools(tools),
                stream=False,
            )
        except OLLAMA_RESPONSE_ERRORS as error:
            raise map_ollama_response_error(error) from error
        except OLLAMA_TRANSPORT_ERRORS as error:
            raise ProviderUnavailableError("Ollama provider unavailable") from error
        return _normalize_tool_response(raw)

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        try:
            raw_stream = await self._client.chat(
                model=request.model,
                messages=self._serialize_messages(request),
                options={"temperature": request.temperature},
                stream=True,
            )
            async for raw_chunk in raw_stream:
                chunk = self._normalize_stream_chunk(raw_chunk)
                if chunk is not None:
                    yield chunk
        except OLLAMA_RESPONSE_ERRORS as error:
            raise map_ollama_response_error(error) from error
        except OLLAMA_TRANSPORT_ERRORS as error:
            raise ProviderUnavailableError("Ollama provider unavailable") from error

    @staticmethod
    def _serialize_messages(request: ChatRequest) -> list[dict[str, str]]:
        return [{"role": message.role, "content": message.content} for message in request.messages]

    @staticmethod
    def _normalize_response(raw_response: Any) -> ChatResponse:
        try:
            input_tokens = getattr(raw_response, "prompt_eval_count", None)
            output_tokens = getattr(raw_response, "eval_count", None)
            usage = OllamaProvider._normalize_usage(input_tokens, output_tokens)
            return ChatResponse(
                content=raw_response.message.content or "",
                model=raw_response.model,
                provider="ollama",
                usage=usage,
                finish_reason=getattr(raw_response, "done_reason", None),
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise ProviderError("Ollama returned an invalid chat response") from error

    @staticmethod
    def _normalize_usage(input_tokens: int | None, output_tokens: int | None) -> Usage | None:
        if input_tokens is None and output_tokens is None:
            return None
        if input_tokens is None or output_tokens is None:
            raise ProviderError("Ollama returned incomplete token usage")
        return Usage(input_tokens=input_tokens, output_tokens=output_tokens)

    @staticmethod
    def _normalize_stream_chunk(raw_chunk: Any) -> ChatStreamChunk | None:
        try:
            content_delta = raw_chunk.message.content or ""
            is_done = raw_chunk.done
            finish_reason = raw_chunk.done_reason if is_done else None
            usage = OllamaProvider._normalize_usage(
                getattr(raw_chunk, "prompt_eval_count", None),
                getattr(raw_chunk, "eval_count", None),
            )
            if not content_delta and finish_reason is None and usage is None:
                return None
            return ChatStreamChunk(
                content_delta=content_delta,
                model=raw_chunk.model,
                provider="ollama",
                finish_reason=finish_reason,
                usage=usage,
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise ProviderError("Ollama returned an invalid chat stream chunk") from error


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
        raw_calls = getattr(raw.message, "tool_calls", None) or []
        calls = tuple(
            ToolCall(getattr(call, "id", None), call.function.name, dict(call.function.arguments)) for call in raw_calls
        )
        return ToolCallingResponse(
            raw.message.content or None,
            calls,
            raw.model,
            "ollama",
            OllamaProvider._normalize_usage(getattr(raw, "prompt_eval_count", None), getattr(raw, "eval_count", None)),
            getattr(raw, "done_reason", None),
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise ProviderError("Ollama returned an invalid tool response") from error
