"""Provider-neutral contracts for native model adapters.

Provider adapters stop at typed in-memory events.  ``OpenAIResponseNormalizer``
is the outward protocol boundary shared by non-streaming and SSE routes.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import Enum
import json
import time
import uuid
from typing import Protocol, TypeAlias

try:
    from backend.chat_contracts import ChatCompletionRequest
except ImportError:  # pragma: no cover - top-level Docker import compatibility
    from chat_contracts import ChatCompletionRequest


class FinishCategory(str, Enum):
    STOP = "stop"
    TOOL_CALLS = "tool_calls"
    LENGTH = "length"
    SAFETY = "safety"
    CANCELLED = "cancelled"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class ProviderMetadata:
    """Sanitized response metadata; never contains opaque provider values."""

    provider: str
    candidate_count: int
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class TextEvent:
    text: str


@dataclass(frozen=True, slots=True)
class FunctionCallEvent:
    """A provider-neutral call with a response-local, non-provider identifier."""

    index: int
    tool_call_id: str
    name: str
    arguments: dict[str, object]


@dataclass(frozen=True, slots=True)
class FinishEvent:
    category: FinishCategory
    native_reason: str
    metadata: ProviderMetadata


ProviderEvent: TypeAlias = TextEvent | FunctionCallEvent | FinishEvent


@dataclass(frozen=True, slots=True)
class ProviderResult:
    events: tuple[TextEvent | FunctionCallEvent, ...]
    finish: FinishEvent


class ProviderAdapter(Protocol):
    def complete(self, request: ChatCompletionRequest) -> AsyncIterator[ProviderEvent]: ...


class ProviderError(RuntimeError):
    """Safe, bounded provider error suitable for later API normalization."""

    code = "PROVIDER_ERROR"
    status_code = 502

    def __init__(self, message: str = "Provider request failed.") -> None:
        super().__init__(message)


class InvalidProviderRequest(ProviderError):
    code = "INVALID_TOOL_CALL"
    status_code = 422


class MissingContinuationState(InvalidProviderRequest):
    code = "STATE_EXPIRED"
    status_code = 409


class ContinuationMismatch(InvalidProviderRequest):
    code = "INVALID_TOOL_CALL"


class ProviderTimeout(ProviderError):
    code = "PROVIDER_TIMEOUT"
    status_code = 504


class ProviderRateLimited(ProviderError):
    code = "RATE_LIMITED"
    status_code = 429


class ProviderUnavailable(ProviderError):
    code = "PROVIDER_UNAVAILABLE"
    status_code = 502


class MalformedProviderResponse(ProviderError):
    code = "PROVIDER_INCOMPATIBLE"
    status_code = 502


class OpenAINormalizationError(InvalidProviderRequest):
    """Raised when neutral events cannot form one bounded OpenAI response."""

    def __init__(self, message: str = "Provider output could not be normalized.") -> None:
        super().__init__(message)


class OpenAIResponseNormalizer:
    """Normalize provider-neutral events without serializing private metadata.

    Streaming consumes one event at a time, so ``StreamingResponse`` applies
    natural backpressure and cancellation propagates to the provider iterator.
    Aggregate/event/tool bounds also protect the non-streaming path.
    """

    def __init__(
        self,
        *,
        max_events: int = 4096,
        max_text_chars: int = 2 * 1024 * 1024,
        max_tool_calls: int = 32,
        max_arguments_chars: int = 256 * 1024,
        argument_fragment_chars: int = 64,
    ) -> None:
        if min(
            max_events,
            max_text_chars,
            max_tool_calls,
            max_arguments_chars,
            argument_fragment_chars,
        ) <= 0:
            raise ValueError("Normalizer bounds must be positive.")
        self.max_events = max_events
        self.max_text_chars = max_text_chars
        self.max_tool_calls = max_tool_calls
        self.max_arguments_chars = max_arguments_chars
        self.argument_fragment_chars = argument_fragment_chars

    @staticmethod
    def new_response_id() -> str:
        return f"chatcmpl-{uuid.uuid4().hex}"

    @staticmethod
    def _finish_reason(category: FinishCategory) -> str:
        return {
            FinishCategory.STOP: "stop",
            FinishCategory.TOOL_CALLS: "tool_calls",
            FinishCategory.LENGTH: "length",
            FinishCategory.SAFETY: "content_filter",
            FinishCategory.CANCELLED: "stop",
            FinishCategory.OTHER: "stop",
        }[category]

    def _arguments(self, event: FunctionCallEvent) -> str:
        if not isinstance(event.arguments, dict):
            raise OpenAINormalizationError("Function arguments must be a JSON object.")
        try:
            rendered = json.dumps(
                event.arguments, ensure_ascii=False, separators=(",", ":"), sort_keys=True
            )
            parsed = json.loads(rendered)
        except (TypeError, ValueError, json.JSONDecodeError):
            raise OpenAINormalizationError("Function arguments are malformed.") from None
        if not isinstance(parsed, dict) or len(rendered) > self.max_arguments_chars:
            raise OpenAINormalizationError("Function arguments exceeded the configured bound.")
        return rendered

    def validate_first_event(self, event: ProviderEvent) -> None:
        """Validate data that must fail before SSE response headers are sent."""

        if isinstance(event, FunctionCallEvent):
            self._arguments(event)
        elif not isinstance(event, (TextEvent, FinishEvent)):
            raise OpenAINormalizationError("Provider emitted an unsupported event.")

    @staticmethod
    def _chunk(
        response_id: str,
        model: str,
        created: int,
        delta: dict[str, object],
        finish_reason: str | None = None,
    ) -> dict[str, object]:
        return {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish_reason,
                }
            ],
        }

    @staticmethod
    def _sse(payload: dict[str, object] | str) -> bytes:
        body = payload if isinstance(payload, str) else json.dumps(
            payload, ensure_ascii=False, separators=(",", ":")
        )
        return f"data: {body}\n\n".encode("utf-8")

    async def non_stream(
        self,
        events: AsyncIterator[ProviderEvent],
        *,
        model: str,
        response_id: str | None = None,
        created: int | None = None,
    ) -> dict[str, object]:
        response_id = response_id or self.new_response_id()
        created = int(time.time()) if created is None else created
        text: list[str] = []
        tools: list[dict[str, object]] = []
        text_chars = 0
        finish: FinishEvent | None = None
        count = 0
        async for event in events:
            count += 1
            if count > self.max_events or finish is not None:
                raise OpenAINormalizationError("Provider emitted an invalid event sequence.")
            if isinstance(event, TextEvent):
                text_chars += len(event.text)
                if text_chars > self.max_text_chars:
                    raise OpenAINormalizationError("Provider text exceeded the configured bound.")
                text.append(event.text)
            elif isinstance(event, FunctionCallEvent):
                if len(tools) >= self.max_tool_calls:
                    raise OpenAINormalizationError("Provider emitted too many function calls.")
                tools.append(
                    {
                        "id": event.tool_call_id,
                        "type": "function",
                        "function": {"name": event.name, "arguments": self._arguments(event)},
                    }
                )
            elif isinstance(event, FinishEvent):
                finish = event
            else:
                raise OpenAINormalizationError("Provider emitted an unsupported event.")
        if finish is None:
            raise OpenAINormalizationError("Provider response ended without a finish event.")
        if tools and finish.category is not FinishCategory.TOOL_CALLS:
            raise OpenAINormalizationError("Function calls require a tool_calls finish reason.")
        if not tools and finish.category is FinishCategory.TOOL_CALLS:
            raise OpenAINormalizationError("tool_calls finish requires a function call.")
        message: dict[str, object] = {
            "role": "assistant",
            "content": "".join(text) if text else None,
        }
        if tools:
            message["tool_calls"] = tools
        metadata = finish.metadata
        return {
            "id": response_id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": self._finish_reason(finish.category),
                }
            ],
            "usage": {
                "prompt_tokens": metadata.prompt_tokens or 0,
                "completion_tokens": metadata.completion_tokens or 0,
                "total_tokens": metadata.total_tokens or 0,
            },
        }

    async def stream(
        self,
        events: AsyncIterator[ProviderEvent],
        *,
        model: str,
        response_id: str | None = None,
        created: int | None = None,
    ) -> AsyncIterator[bytes]:
        response_id = response_id or self.new_response_id()
        created = int(time.time()) if created is None else created
        count = 0
        text_chars = 0
        tool_indexes: dict[str, int] = {}
        finish: FinishEvent | None = None
        yield self._sse(self._chunk(response_id, model, created, {"role": "assistant"}))
        async for event in events:
            count += 1
            if count > self.max_events or finish is not None:
                raise OpenAINormalizationError("Provider emitted an invalid event sequence.")
            if isinstance(event, TextEvent):
                text_chars += len(event.text)
                if text_chars > self.max_text_chars:
                    raise OpenAINormalizationError("Provider text exceeded the configured bound.")
                yield self._sse(
                    self._chunk(response_id, model, created, {"content": event.text})
                )
            elif isinstance(event, FunctionCallEvent):
                if event.tool_call_id in tool_indexes:
                    raise OpenAINormalizationError("Function call identity was repeated.")
                if len(tool_indexes) >= self.max_tool_calls:
                    raise OpenAINormalizationError("Provider emitted too many function calls.")
                # Provider indexes are advisory. Missing/invalid/colliding values
                # become stable response-local indexes in arrival order.
                proposed = event.index
                used = set(tool_indexes.values())
                index = proposed if isinstance(proposed, int) and proposed >= 0 and proposed not in used else len(used)
                tool_indexes[event.tool_call_id] = index
                arguments = self._arguments(event)
                yield self._sse(
                    self._chunk(
                        response_id,
                        model,
                        created,
                        {
                            "tool_calls": [
                                {
                                    "index": index,
                                    "id": event.tool_call_id,
                                    "type": "function",
                                    "function": {"name": event.name, "arguments": ""},
                                }
                            ]
                        },
                    )
                )
                for start in range(0, len(arguments), self.argument_fragment_chars):
                    yield self._sse(
                        self._chunk(
                            response_id,
                            model,
                            created,
                            {
                                "tool_calls": [
                                    {
                                        "index": index,
                                        "function": {
                                            "arguments": arguments[
                                                start : start + self.argument_fragment_chars
                                            ]
                                        },
                                    }
                                ]
                            },
                        )
                    )
            elif isinstance(event, FinishEvent):
                finish = event
            else:
                raise OpenAINormalizationError("Provider emitted an unsupported event.")
        if finish is None:
            raise OpenAINormalizationError("Provider response ended without a finish event.")
        if tool_indexes and finish.category is not FinishCategory.TOOL_CALLS:
            raise OpenAINormalizationError("Function calls require a tool_calls finish reason.")
        if not tool_indexes and finish.category is FinishCategory.TOOL_CALLS:
            raise OpenAINormalizationError("tool_calls finish requires a function call.")
        yield self._sse(
            self._chunk(
                response_id,
                model,
                created,
                {},
                self._finish_reason(finish.category),
            )
        )
        yield self._sse("[DONE]")
