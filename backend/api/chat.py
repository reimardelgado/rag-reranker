"""OpenAI-compatible text chat routes.

The adapter receives runtime dependencies from ``app.py`` so this module stays
free of infrastructure initialization and can be extended without creating an
import cycle with the composition root.
"""

import json
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Awaitable, Callable

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

try:
    from backend.chat_contracts import (
        ChatCompletionRequest,
        OpenAIChatRequest,
        extract_user_query,
        textual_content_parts,
    )
    from backend.providers.base import (
        FinishCategory,
        FinishEvent,
        FunctionCallEvent,
        InvalidProviderRequest,
        OpenAINormalizationError,
        OpenAIResponseNormalizer,
        ProviderAdapter,
        ProviderError,
        ProviderEvent,
        TextEvent,
    )
    from backend.grounding.provenance import ImageProvenance, render_adjacent_provenance
    from backend.grounding.validation import ValidatedGrounding
except ImportError:  # Docker runs backend/app.py as top-level ``app``.
    from chat_contracts import (
        ChatCompletionRequest,
        OpenAIChatRequest,
        extract_user_query,
        textual_content_parts,
    )
    from providers.base import (
        FinishCategory,
        FinishEvent,
        FunctionCallEvent,
        InvalidProviderRequest,
        OpenAINormalizationError,
        OpenAIResponseNormalizer,
        ProviderAdapter,
        ProviderError,
        ProviderEvent,
        TextEvent,
    )
    from grounding.provenance import ImageProvenance, render_adjacent_provenance
    from grounding.validation import ValidatedGrounding


IMAGE_TOOL_NAME = "generate_image"
_IMAGE_OPTION_NAMES = frozenset({"size", "quality", "style"})


class UnsupportedImageTool(ProviderError):
    """The client did not advertise a usable image-generation function."""

    code = "UNSUPPORTED_TOOL"
    status_code = 409


class InvalidImageToolCall(InvalidProviderRequest):
    """A forced image call failed the final fail-closed boundary."""

    code = "INVALID_TOOL_CALL"
    status_code = 422


@dataclass(frozen=True, slots=True)
class ImageToolEnforcementLimits:
    max_buffered_events: int = 32
    max_arguments_chars: int = 256 * 1024

    def __post_init__(self) -> None:
        if min(self.max_buffered_events, self.max_arguments_chars) <= 0:
            raise ValueError("Image-tool enforcement bounds must be positive.")


class ImageToolEnforcer:
    """Prepare and validate one grounded ``generate_image`` provider turn.

    The boundary is intentionally inactive until explicitly composed by a
    later rollout task.  It buffers the complete provider turn so no text or
    tool call can escape before the function name, finish reason, arguments,
    schema, and configured bounds have all passed validation.
    """

    def __init__(
        self,
        request: ChatCompletionRequest,
        grounding: ValidatedGrounding,
        *,
        limits: ImageToolEnforcementLimits | None = None,
    ) -> None:
        self._grounding = grounding.model_copy(deep=True)
        self._limits = limits or ImageToolEnforcementLimits()
        self._properties, self._required = self._image_schema(request)

    @staticmethod
    def _image_schema(
        request: ChatCompletionRequest,
    ) -> tuple[dict[str, object], frozenset[str]]:
        matches = [
            tool
            for tool in request.tools or ()
            if tool.type == "function"
            and tool.function is not None
            and tool.function.name == IMAGE_TOOL_NAME
        ]
        if len(matches) != 1:
            raise UnsupportedImageTool(
                "A single compatible generate_image tool must be advertised."
            )
        parameters = matches[0].function.parameters
        if not isinstance(parameters, dict) or parameters.get("type") != "object":
            raise UnsupportedImageTool("The generate_image schema is incompatible.")
        properties = parameters.get("properties")
        if not isinstance(properties, dict):
            raise UnsupportedImageTool("The generate_image schema is incompatible.")
        prompt_schema = properties.get("prompt")
        if not isinstance(prompt_schema, dict) or not ImageToolEnforcer._accepts_string(
            prompt_schema, None
        ):
            raise UnsupportedImageTool("The generate_image prompt schema is incompatible.")
        required_value = parameters.get("required", [])
        if (
            not isinstance(required_value, list)
            or any(not isinstance(name, str) for name in required_value)
            or any(name not in properties for name in required_value)
            or any(
                name != "prompt" and name not in _IMAGE_OPTION_NAMES
                for name in required_value
            )
        ):
            raise UnsupportedImageTool("The generate_image required fields are incompatible.")
        return dict(properties), frozenset(required_value)

    @staticmethod
    def _accepts_string(schema: object, value: str | None) -> bool:
        if not isinstance(schema, dict):
            return False
        schema_type = schema.get("type")
        if isinstance(schema_type, list):
            accepts_type = "string" in schema_type
        else:
            accepts_type = schema_type == "string"
        if not accepts_type:
            return False
        if value is None:
            return True
        enum = schema.get("enum")
        if enum is not None and (not isinstance(enum, list) or value not in enum):
            return False
        minimum = schema.get("minLength")
        maximum = schema.get("maxLength")
        if isinstance(minimum, int) and len(value) < minimum:
            return False
        if isinstance(maximum, int) and len(value) > maximum:
            return False
        return True

    def forced_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        """Return a detached request that forces the advertised image function."""

        # Revalidate the supplied request instead of relying on constructor
        # identity; callers cannot switch schemas between preparation and use.
        properties, required = self._image_schema(request)
        if properties != self._properties or required != self._required:
            raise UnsupportedImageTool("The generate_image schema changed during enforcement.")
        payload = request.model_dump(exclude_unset=True)
        payload["tool_choice"] = {
            "type": "function",
            "function": {"name": IMAGE_TOOL_NAME},
        }
        return ChatCompletionRequest.model_validate(payload)

    def _validated_arguments(self, arguments: object) -> dict[str, object]:
        if not isinstance(arguments, dict):
            raise InvalidImageToolCall("Image-tool arguments must be a JSON object.")
        try:
            rendered = json.dumps(
                arguments, ensure_ascii=False, separators=(",", ":"), sort_keys=True
            )
            parsed = json.loads(rendered)
        except (TypeError, ValueError, json.JSONDecodeError):
            raise InvalidImageToolCall("Image-tool arguments are malformed.") from None
        if len(rendered.encode("utf-8")) > self._limits.max_arguments_chars:
            raise InvalidImageToolCall("Image-tool arguments exceeded the configured bound.")
        if not isinstance(parsed, dict) or "prompt" not in parsed:
            raise InvalidImageToolCall("Image-tool arguments require a prompt.")
        if not isinstance(parsed["prompt"], str):
            raise InvalidImageToolCall("Image-tool prompt must be text.")

        unsupported = set(parsed) - {"prompt"} - _IMAGE_OPTION_NAMES
        undeclared = set(parsed) - set(self._properties)
        if unsupported or undeclared:
            raise InvalidImageToolCall("Image-tool arguments contain unsupported fields.")

        output: dict[str, object] = {
            "prompt": self._grounding.image_spec.canonical_prompt
        }
        for name in _IMAGE_OPTION_NAMES:
            provider_value = parsed.get(name)
            grounded_value = getattr(self._grounding.image_spec, name)
            if provider_value is not None:
                if not isinstance(provider_value, str):
                    raise InvalidImageToolCall("Image-tool option is not schema-supported.")
                if not self._accepts_string(self._properties[name], provider_value):
                    raise InvalidImageToolCall("Image-tool option violates its schema.")
            value = grounded_value if grounded_value is not None else provider_value
            if value is None:
                continue
            if name not in self._properties or not isinstance(value, str):
                raise InvalidImageToolCall("Image-tool option is not schema-supported.")
            if not self._accepts_string(self._properties[name], value):
                raise InvalidImageToolCall("Image-tool option violates its schema.")
            output[name] = value

        if not self._accepts_string(self._properties["prompt"], output["prompt"]):
            raise InvalidImageToolCall("Canonical prompt violates the advertised schema.")
        if not self._required.issubset(output):
            raise InvalidImageToolCall("Image-tool required arguments are missing.")
        try:
            output_size = len(
                json.dumps(
                    output, ensure_ascii=False, separators=(",", ":"), sort_keys=True
                ).encode("utf-8")
            )
        except (TypeError, ValueError):  # Defensive: grounding models are typed.
            raise InvalidImageToolCall("Validated image arguments are malformed.") from None
        if output_size > self._limits.max_arguments_chars:
            raise InvalidImageToolCall("Validated image arguments exceeded the configured bound.")
        return output

    async def enforce(
        self, events: AsyncIterator[ProviderEvent]
    ) -> AsyncIterator[ProviderEvent]:
        """Buffer and release exactly one validated image call and its finish."""

        buffered: list[ProviderEvent] = []
        async for event in events:
            buffered.append(event)
            if len(buffered) > self._limits.max_buffered_events:
                raise InvalidImageToolCall("Image-tool response exceeded the event bound.")

        calls = [event for event in buffered if isinstance(event, FunctionCallEvent)]
        finishes = [event for event in buffered if isinstance(event, FinishEvent)]
        if any(isinstance(event, TextEvent) for event in buffered):
            raise InvalidImageToolCall("Forced image responses cannot contain assistant text.")
        if len(calls) != 1 or len(finishes) != 1 or buffered[-1:] != finishes:
            raise InvalidImageToolCall("Image-tool response sequence is invalid.")
        if len(buffered) != 2:
            raise InvalidImageToolCall("Image-tool response contains unsupported events.")
        call = calls[0]
        finish = finishes[0]
        if call.name != IMAGE_TOOL_NAME:
            raise InvalidImageToolCall("Provider selected an unexpected function.")
        if finish.category is not FinishCategory.TOOL_CALLS:
            raise InvalidImageToolCall("Image-tool response has an invalid finish reason.")

        yield FunctionCallEvent(
            index=call.index,
            tool_call_id=call.tool_call_id,
            name=IMAGE_TOOL_NAME,
            arguments=self._validated_arguments(call.arguments),
        )
        yield finish


@dataclass(frozen=True, slots=True)
class ImageToolContinuationLimits:
    max_buffered_events: int = 4096
    max_text_chars: int = 2 * 1024 * 1024
    max_sidecar_chars: int = 64 * 1024

    def __post_init__(self) -> None:
        if min(
            self.max_buffered_events,
            self.max_text_chars,
            self.max_sidecar_chars,
        ) <= 0:
            raise ValueError("Image-tool continuation bounds must be positive.")


class ImageToolContinuation:
    """Validate one tool result and append provenance to the final turn.

    Like ``ImageToolEnforcer``, this contract is intentionally inactive.  It
    buffers the final provider turn before releasing text so malformed or
    recursive continuations fail closed without a partial unlabelled result.
    """

    def __init__(
        self,
        tool_call_id: str,
        provenance: ImageProvenance,
        *,
        limits: ImageToolContinuationLimits | None = None,
    ) -> None:
        if not isinstance(tool_call_id, str) or not tool_call_id.strip():
            raise InvalidImageToolCall("Image continuation requires a tool-call ID.")
        self._tool_call_id = tool_call_id
        self._provenance = provenance.model_copy(deep=True)
        self._limits = limits or ImageToolContinuationLimits()

    def continuation_request(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionRequest:
        """Return a detached request after validating the adjacent public pair."""

        if len(request.messages) < 2:
            raise InvalidImageToolCall("Image continuation messages are incomplete.")
        assistant = request.messages[-2]
        tool_result = request.messages[-1]
        if assistant.role != "assistant" or tool_result.role != "tool":
            raise InvalidImageToolCall("Image continuation messages are out of order.")
        calls = assistant.tool_calls or ()
        if len(calls) != 1:
            raise InvalidImageToolCall("Image continuation requires one tool call.")
        call = calls[0]
        if (
            call.id != self._tool_call_id
            or call.function.name != IMAGE_TOOL_NAME
            or tool_result.tool_call_id != self._tool_call_id
        ):
            raise InvalidImageToolCall("Image continuation does not match its tool call.")
        return request.model_copy(deep=True)

    async def append_provenance(
        self, events: AsyncIterator[ProviderEvent]
    ) -> AsyncIterator[ProviderEvent]:
        """Release a valid final assistant turn with adjacent public provenance."""

        buffered: list[ProviderEvent] = []
        text_chars = 0
        async for event in events:
            buffered.append(event)
            if len(buffered) > self._limits.max_buffered_events:
                raise InvalidImageToolCall("Image continuation exceeded the event bound.")
            if isinstance(event, TextEvent):
                text_chars += len(event.text)
                if text_chars > self._limits.max_text_chars:
                    raise InvalidImageToolCall("Image continuation text exceeded its bound.")

        finishes = [event for event in buffered if isinstance(event, FinishEvent)]
        texts = [event for event in buffered if isinstance(event, TextEvent)]
        if (
            len(finishes) != 1
            or buffered[-1:] != finishes
            or len(texts) + 1 != len(buffered)
            or finishes[0].category is not FinishCategory.STOP
            or not any(event.text for event in texts)
        ):
            raise InvalidImageToolCall("Image continuation response sequence is invalid.")

        for event in texts:
            yield event
        yield TextEvent(
            render_adjacent_provenance(
                self._provenance,
                max_sidecar_chars=self._limits.max_sidecar_chars,
            )
        )
        yield finishes[0]


@dataclass(frozen=True)
class ChatDependencies:
    get_gemini_models: Callable[[], list]
    get_ollama_models: Callable[[], Awaitable[list]]
    retrieve_blocks: Callable[[str, int], list]
    is_ollama_model: Callable[[str], bool]
    call_ollama_chat: Callable[[str, list, float], Awaitable[str]]
    call_gemini_chat: Callable[[str, str, float], Awaitable[str]]
    logger: object
    provider_adapter: ProviderAdapter | None = None
    response_normalizer: OpenAIResponseNormalizer | None = None


@dataclass(frozen=True)
class ChatRoutes:
    router: APIRouter
    list_models: Callable
    openai_chat_completions: Callable


def build_rag_system_prompt(rag_blocks: list) -> str:
    context = "\n\n".join(rag_blocks)
    return (
        "Eres un asistente experto en arqueología ibera. "
        "Responde usando principalmente el siguiente contexto documental. "
        "Si la pregunta no está cubierta, indícalo.\n\n"
        f"=== CONTEXTO ===\n{context}\n=== FIN CONTEXTO ==="
    )


def openai_response(model: str, content: str) -> dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _openai_error(error: ProviderError, request_id: str) -> dict[str, object]:
    return {
        "error": {
            "message": str(error),
            "type": "provider_error",
            "code": error.code,
        },
        "request_id": request_id,
    }


async def _prepend_event(
    first: ProviderEvent, remaining: AsyncIterator[ProviderEvent]
) -> AsyncIterator[ProviderEvent]:
    yield first
    async for event in remaining:
        yield event


def _adapter_request(
    request: ChatCompletionRequest, rag_blocks: list
) -> ChatCompletionRequest:
    """Add RAG context while retaining every client field and message payload."""

    if not rag_blocks:
        return request.model_copy(deep=True)
    payload = request.model_dump(exclude_unset=True)
    messages = [message.model_dump(exclude_unset=True) for message in request.messages]
    payload["messages"] = [
        {"role": "system", "content": build_rag_system_prompt(rag_blocks)},
        *messages,
    ]
    return ChatCompletionRequest.model_validate(payload)


async def _close_iterator(iterator: object) -> None:
    close = getattr(iterator, "aclose", None)
    if close is not None:
        await close()


def create_chat_router(
    dependencies: ChatDependencies,
    *,
    default_model: str,
    default_top_k: int,
) -> ChatRoutes:
    # Preserve the existing request schema defaults while keeping configuration
    # ownership in the composition root.
    OpenAIChatRequest.model_fields["model"].default = default_model
    OpenAIChatRequest.model_fields["top_k"].default = default_top_k
    OpenAIChatRequest.model_rebuild(force=True)

    router = APIRouter()

    @router.get("/v1/models")
    @router.get("/models")
    async def list_models():
        return {
            "object": "list",
            "data": dependencies.get_gemini_models()
            + await dependencies.get_ollama_models(),
        }

    @router.post("/v1/chat/completions")
    @router.post("/chat/completions")
    async def openai_chat_completions(req: ChatCompletionRequest):
        query = extract_user_query(req.messages)
        if not query:
            raise HTTPException(
                status_code=400, detail="No se encontró mensaje de usuario."
            )

        rag_blocks = dependencies.retrieve_blocks(query, req.top_k)

        if dependencies.provider_adapter is not None:
            normalizer = dependencies.response_normalizer or OpenAIResponseNormalizer()
            provider_request = _adapter_request(req, rag_blocks)
            provider_events = dependencies.provider_adapter.complete(provider_request)
            response_id = normalizer.new_response_id()
            try:
                if req.stream:
                    try:
                        first_event = await anext(provider_events)
                    except StopAsyncIteration:
                        raise OpenAINormalizationError(
                            "Provider response ended without events."
                        ) from None
                    normalizer.validate_first_event(first_event)
                    normalized = normalizer.stream(
                        _prepend_event(first_event, provider_events),
                        model=req.model,
                        response_id=response_id,
                    )
                    first_chunk = await anext(normalized)

                    async def stream_body() -> AsyncIterator[bytes]:
                        try:
                            yield first_chunk
                            async for chunk in normalized:
                                yield chunk
                        except ProviderError as error:
                            yield normalizer._sse(_openai_error(error, response_id))
                            yield normalizer._sse("[DONE]")
                        finally:
                            await _close_iterator(normalized)
                            await _close_iterator(provider_events)

                    return StreamingResponse(stream_body(), media_type="text/event-stream")

                return await normalizer.non_stream(
                    provider_events, model=req.model, response_id=response_id
                )
            except ProviderError as error:
                await _close_iterator(provider_events)
                return JSONResponse(
                    status_code=error.status_code,
                    content=_openai_error(error, response_id),
                )
            except BaseException:
                # Cancellation and unexpected failures are never converted into
                # successful completions; release the provider iterator and
                # preserve the original exception semantics.
                await _close_iterator(provider_events)
                raise

        forwarded_messages = [
            message.model_dump(exclude_unset=True) for message in req.messages
        ]
        enriched = []
        has_system = any(
            message.get("role") == "system" for message in forwarded_messages
        )
        if rag_blocks:
            rag_system = build_rag_system_prompt(rag_blocks)
            if has_system:
                for message in forwarded_messages:
                    if message.get("role") == "system":
                        enriched.append(
                            {
                                "role": "system",
                                "content": message["content"] + "\n\n" + rag_system,
                            }
                        )
                    else:
                        enriched.append(message)
            else:
                enriched = [{"role": "system", "content": rag_system}] + list(
                    forwarded_messages
                )
        else:
            enriched = list(forwarded_messages)

        try:
            if dependencies.is_ollama_model(req.model):
                answer_text = await dependencies.call_ollama_chat(
                    req.model, enriched, req.temperature
                )
            else:
                prompt_parts = []
                for message in enriched:
                    role = message.get("role", "user")
                    content = message.get("content", "")
                    if isinstance(content, list):
                        content = " ".join(textual_content_parts(content))
                    prefix = {"system": "[Sistema]", "assistant": "[Asistente]"}.get(
                        role, "[Usuario]"
                    )
                    prompt_parts.append(f"{prefix}: {content}")
                answer_text = await dependencies.call_gemini_chat(
                    req.model, "\n\n".join(prompt_parts), req.temperature
                )
        except Exception as exc:
            dependencies.logger.exception("[bridge] Error LLM")
            raise HTTPException(
                status_code=502,
                detail=f"Error LLM: {type(exc).__name__}: {exc}",
            )

        if rag_blocks:
            sources, seen_sources = [], set()
            for block in rag_blocks:
                if block.startswith("[") and "]" in block:
                    tag = block.split("]")[0] + "]"
                    if tag not in seen_sources:
                        seen_sources.add(tag)
                        sources.append(tag)
            if sources:
                answer_text += "\n\n---\n**Fuentes**\n" + "\n".join(
                    "- " + source for source in sources
                )

        return openai_response(req.model, answer_text)

    return ChatRoutes(
        router=router,
        list_models=list_models,
        openai_chat_completions=openai_chat_completions,
    )
