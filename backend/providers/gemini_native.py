"""Native Gemini generateContent adapter.

The adapter is intentionally inactive: no production route imports or composes
it yet.  It translates typed OpenAI-style requests to native Gemini JSON and
returns provider-neutral in-memory events.  OpenAI SSE/non-stream serialization
is reserved for task 6.2.
"""

from __future__ import annotations

import base64
import binascii
import json
import re
import time
import uuid
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote

import httpx

try:
    from backend.chat_contracts import ChatCompletionRequest
    from backend.providers.base import (
        ContinuationMismatch,
        FinishCategory,
        FinishEvent,
        FunctionCallEvent,
        InvalidProviderRequest,
        MalformedProviderResponse,
        MissingContinuationState,
        ProviderEvent,
        ProviderMetadata,
        ProviderRateLimited,
        ProviderResult,
        ProviderTimeout,
        ProviderUnavailable,
        TextEvent,
    )
except ImportError:  # pragma: no cover - top-level Docker import compatibility
    from chat_contracts import ChatCompletionRequest
    from providers.base import (
        ContinuationMismatch,
        FinishCategory,
        FinishEvent,
        FunctionCallEvent,
        InvalidProviderRequest,
        MalformedProviderResponse,
        MissingContinuationState,
        ProviderEvent,
        ProviderMetadata,
        ProviderRateLimited,
        ProviderResult,
        ProviderTimeout,
        ProviderUnavailable,
        TextEvent,
    )


_DATA_URL = re.compile(
    r"^data:(?P<mime>[a-zA-Z0-9][a-zA-Z0-9.+-]*/[a-zA-Z0-9][a-zA-Z0-9.+-]*);base64,(?P<data>[A-Za-z0-9+/=\r\n]+)$"
)
_SAFE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.:-]{0,127}$")


@dataclass(frozen=True, slots=True)
class _NativeCall:
    public_id: str
    name: str
    arguments: dict[str, object]
    provider_id: str | None = field(default=None, repr=False)


@dataclass(frozen=True, slots=True)
class _PendingContinuation:
    calls: tuple[_NativeCall, ...]
    model_content: dict[str, object] = field(repr=False)
    expires_at: float = float("inf")


class GeminiNativeAdapter:
    """Dependency-injected native Gemini adapter with no implicit retries."""

    def __init__(
        self,
        *,
        api_key: str,
        client: httpx.AsyncClient,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        timeout: httpx.Timeout | float = httpx.Timeout(30.0, connect=5.0),
        max_request_bytes: int = 4 * 1024 * 1024,
        max_response_bytes: int = 2 * 1024 * 1024,
        max_inline_bytes: int = 10 * 1024 * 1024,
        max_pending_calls: int = 32,
        pending_ttl_seconds: float = 600.0,
        id_factory: Callable[[], str] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not api_key:
            raise ValueError("Gemini API key is required.")
        if (
            max_request_bytes <= 0
            or max_response_bytes <= 0
            or max_inline_bytes <= 0
            or max_pending_calls <= 0
            or not pending_ttl_seconds > 0
        ):
            raise ValueError("Adapter bounds must be positive.")
        self._api_key = api_key
        self._client = client
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_request_bytes = max_request_bytes
        self._max_response_bytes = max_response_bytes
        self._max_inline_bytes = max_inline_bytes
        self._max_pending_calls = max_pending_calls
        self._id_factory = id_factory or (lambda: f"call_{uuid.uuid4().hex}")
        self._pending_ttl_seconds = float(pending_ttl_seconds)
        self._clock = clock
        self._pending_by_id: dict[str, _PendingContinuation] = {}

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(base_url={self._base_url!r}, "
            f"max_response_bytes={self._max_response_bytes!r})"
        )

    async def complete(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[ProviderEvent]:
        result = await self.generate(request)
        for event in result.events:
            yield event
        yield result.finish

    @property
    def pending_continuation_count(self) -> int:
        self._purge_expired_pending()
        return len(self._pending_by_id)

    def discard_continuation(self, tool_call_id: str) -> bool:
        """Forget the private continuation that owns ``tool_call_id``.

        Used when a function call is consumed internally (for example a
        structured-output call) and will never be continued by a client.
        """

        pending = self._pending_by_id.get(tool_call_id)
        if pending is None:
            return False
        self._discard_pending(pending)
        return True

    def rebind_continuation_arguments(
        self, tool_call_id: str, arguments: Mapping[str, object]
    ) -> None:
        """Accept validated replacement arguments for one pending call.

        A fail-closed enforcement boundary may replace provider-chosen
        arguments (for example with a canonical grounded prompt) before the
        call reaches the client.  The client then echoes the replacement, so
        continuation matching must expect it.  The private native model
        content, including the thought signature, is deliberately unchanged.
        """

        self._purge_expired_pending()
        pending = self._pending_by_id.get(tool_call_id)
        if pending is None:
            raise MissingContinuationState("Required continuation state is missing.")
        try:
            replacement = json.loads(
                json.dumps(dict(arguments), ensure_ascii=False, separators=(",", ":"))
            )
        except (TypeError, ValueError):
            raise InvalidProviderRequest("Replacement arguments are not JSON.") from None
        calls = tuple(
            _NativeCall(
                public_id=call.public_id,
                name=call.name,
                arguments=deepcopy(replacement) if call.public_id == tool_call_id else call.arguments,
                provider_id=call.provider_id,
            )
            for call in pending.calls
        )
        rebound = _PendingContinuation(
            calls=calls,
            model_content=pending.model_content,
            expires_at=pending.expires_at,
        )
        for call in calls:
            self._pending_by_id[call.public_id] = rebound

    def _purge_expired_pending(self) -> None:
        now = self._clock()
        expired = {
            id(pending): pending
            for pending in self._pending_by_id.values()
            if now >= pending.expires_at
        }
        for pending in expired.values():
            self._discard_pending(pending)

    async def generate(self, request: ChatCompletionRequest) -> ProviderResult:
        self._purge_expired_pending()
        body, consumed = self._build_request(request)
        try:
            request_size = len(
                json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            )
        except (TypeError, ValueError):
            raise InvalidProviderRequest("Request contains unsupported JSON values.") from None
        if request_size > self._max_request_bytes:
            raise InvalidProviderRequest("Gemini request exceeded the configured bound.")
        # Continuations are single-use.  Discard before dispatch so cancellation,
        # timeout, provider failure, and replay can never reuse a signature.
        if consumed is not None:
            self._discard_pending(consumed)

        model = quote(request.model, safe="-._~")
        url = f"{self._base_url}/models/{model}:generateContent"
        try:
            response = await self._client.post(
                url,
                headers={"x-goog-api-key": self._api_key, "Content-Type": "application/json"},
                json=body,
                timeout=self._timeout,
            )
        except httpx.TimeoutException:
            raise ProviderTimeout("Gemini request timed out.") from None
        except httpx.RequestError:
            raise ProviderUnavailable("Gemini transport is unavailable.") from None

        if response.status_code == 429:
            raise ProviderRateLimited("Gemini rate limit was reached.")
        if response.status_code >= 500:
            raise ProviderUnavailable("Gemini service is unavailable.")
        if response.status_code >= 400:
            raise InvalidProviderRequest("Gemini rejected the request.")
        if len(response.content) > self._max_response_bytes:
            raise MalformedProviderResponse("Gemini response exceeded the configured bound.")
        try:
            payload = json.loads(response.content)
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise MalformedProviderResponse("Gemini returned malformed JSON.") from None
        if not isinstance(payload, dict):
            raise MalformedProviderResponse("Gemini returned an invalid response envelope.")
        return self._parse_response(payload)

    def _build_request(
        self, request: ChatCompletionRequest
    ) -> tuple[dict[str, object], _PendingContinuation | None]:
        request_data = request.model_dump(exclude_unset=True)
        allowed_fields = {
            "model",
            "messages",
            "temperature",
            "stream",
            "stream_options",
            "tools",
            "tool_choice",
            "top_k",
            "max_tokens",
            "top_p",
            "stop",
            "parallel_tool_calls",
        }
        if set(request_data) - allowed_fields:
            raise InvalidProviderRequest("Request contains unsupported provider fields.")
        if request_data.get("parallel_tool_calls") not in (None, True):
            raise InvalidProviderRequest("Disabling Gemini parallel calls is unsupported.")
        body: dict[str, object] = {}
        contents, system_instruction, consumed = self._convert_messages(request_data["messages"])
        body["contents"] = contents
        if system_instruction:
            body["systemInstruction"] = {"parts": system_instruction}

        tools = request_data.get("tools")
        if tools is not None:
            body["tools"] = [{"functionDeclarations": self._convert_tools(tools)}]
        tool_config = self._convert_tool_choice(request_data.get("tool_choice"), tools)
        if tool_config is not None:
            body["toolConfig"] = {"functionCallingConfig": tool_config}

        generation_config: dict[str, object] = {"temperature": request.temperature}
        supported_options = {
            "max_tokens": "maxOutputTokens",
            "top_p": "topP",
            "stop": "stopSequences",
        }
        for source, target in supported_options.items():
            if source in request_data:
                value = deepcopy(request_data[source])
                if source == "stop" and isinstance(value, str):
                    value = [value]
                generation_config[target] = value
        body["generationConfig"] = generation_config
        return body, consumed

    def _convert_tools(self, tools: object) -> list[dict[str, object]]:
        if not isinstance(tools, list) or not tools:
            raise InvalidProviderRequest("Tools must be a non-empty list.")
        declarations: list[dict[str, object]] = []
        names: set[str] = set()
        for tool in tools:
            if not isinstance(tool, Mapping) or tool.get("type") != "function":
                raise InvalidProviderRequest("Only function tools are supported by Gemini.")
            if set(tool) - {"type", "function"}:
                raise InvalidProviderRequest("Function tool contains unsupported fields.")
            function = tool.get("function")
            if not isinstance(function, Mapping):
                raise InvalidProviderRequest("Function tool declaration is missing.")
            if set(function) - {"name", "description", "parameters", "strict"}:
                raise InvalidProviderRequest("Function declaration contains unsupported fields.")
            name = function.get("name")
            if not isinstance(name, str) or not _SAFE_NAME.fullmatch(name) or name in names:
                raise InvalidProviderRequest("Function tool name is invalid or duplicated.")
            names.add(name)
            declaration: dict[str, object] = {"name": name}
            description = function.get("description")
            if description is not None:
                if not isinstance(description, str):
                    raise InvalidProviderRequest("Function description must be text.")
                declaration["description"] = description
            parameters = function.get("parameters")
            if parameters is not None:
                if not isinstance(parameters, Mapping):
                    raise InvalidProviderRequest("Function parameters must be a JSON schema object.")
                declaration["parameters"] = deepcopy(dict(parameters))
            if function.get("strict") not in (None, False):
                raise InvalidProviderRequest("Strict OpenAI function mode is unsupported by Gemini.")
            declarations.append(declaration)
        return declarations

    def _convert_tool_choice(
        self, choice: object, tools: object
    ) -> dict[str, object] | None:
        if choice is None:
            return None
        if tools is None:
            raise InvalidProviderRequest("Tool choice requires tool declarations.")
        if isinstance(choice, str):
            modes = {"auto": "AUTO", "none": "NONE", "required": "ANY"}
            if choice not in modes:
                raise InvalidProviderRequest("Unsupported tool choice.")
            return {"mode": modes[choice]}
        if isinstance(choice, Mapping):
            function = choice.get("function")
            if choice.get("type") != "function" or not isinstance(function, Mapping):
                raise InvalidProviderRequest("Unsupported tool choice object.")
            name = function.get("name")
            declared = {
                item["function"]["name"]
                for item in tools
                if isinstance(item, Mapping)
                and isinstance(item.get("function"), Mapping)
                and isinstance(item["function"].get("name"), str)
            }
            if not isinstance(name, str) or name not in declared:
                raise InvalidProviderRequest("Tool choice does not match a declaration.")
            return {"mode": "ANY", "allowedFunctionNames": [name]}
        raise InvalidProviderRequest("Unsupported tool choice.")

    def _convert_messages(
        self, messages: object
    ) -> tuple[list[dict[str, object]], list[dict[str, object]], _PendingContinuation | None]:
        if not isinstance(messages, list) or not messages:
            raise InvalidProviderRequest("Messages must be a non-empty list.")
        contents: list[dict[str, object]] = []
        system_parts: list[dict[str, object]] = []
        consumed: _PendingContinuation | None = None
        index = 0
        while index < len(messages):
            message = messages[index]
            if not isinstance(message, Mapping):
                raise InvalidProviderRequest("Message is invalid.")
            role = message.get("role")
            if role == "system":
                if set(message) - {"role", "content"}:
                    raise InvalidProviderRequest("System message contains unsupported fields.")
                if contents:
                    raise InvalidProviderRequest("System messages must precede conversation content.")
                system_parts.extend(self._convert_content(message.get("content"), text_only=True))
            elif role in ("user", "assistant"):
                allowed = (
                    {"role", "content", "tool_calls"}
                    if role == "assistant"
                    else {"role", "content"}
                )
                if set(message) - allowed:
                    raise InvalidProviderRequest("Message contains unsupported fields.")
                tool_calls = message.get("tool_calls") if role == "assistant" else None
                if tool_calls:
                    if consumed is not None:
                        raise ContinuationMismatch("Only one pending continuation is supported per request.")
                    pending = self._match_pending(tool_calls)
                    contents.append(deepcopy(pending.model_content))
                    response_parts: list[dict[str, object]] = []
                    for expected in pending.calls:
                        index += 1
                        if index >= len(messages):
                            raise ContinuationMismatch("Every function call requires one tool response.")
                        response = messages[index]
                        response_parts.append(self._convert_tool_response(response, expected))
                    contents.append({"role": "user", "parts": response_parts})
                    consumed = pending
                else:
                    native_role = "model" if role == "assistant" else "user"
                    parts = self._convert_content(message.get("content"))
                    if not parts:
                        raise InvalidProviderRequest("Message content cannot be empty.")
                    contents.append({"role": native_role, "parts": parts})
            elif role == "tool":
                raise ContinuationMismatch("Tool response has no matching assistant function call.")
            else:
                raise InvalidProviderRequest("Unsupported message role.")
            index += 1
        return contents, system_parts, consumed

    def _convert_content(
        self, content: object, *, text_only: bool = False
    ) -> list[dict[str, object]]:
        if isinstance(content, str):
            return [{"text": content}] if content else []
        if not isinstance(content, list):
            raise InvalidProviderRequest("Message content must be text or multimodal parts.")
        parts: list[dict[str, object]] = []
        for part in content:
            if not isinstance(part, Mapping):
                raise InvalidProviderRequest("Multimodal content part is invalid.")
            part_type = part.get("type")
            if part_type == "text":
                if set(part) - {"type", "text"}:
                    raise InvalidProviderRequest("Text part contains unsupported fields.")
                text = part.get("text")
                if not isinstance(text, str):
                    raise InvalidProviderRequest("Text part is missing text.")
                parts.append({"text": text})
                continue
            if text_only:
                raise InvalidProviderRequest("System instructions only support text parts.")
            if part_type == "image_url":
                if set(part) - {"type", "image_url"}:
                    raise InvalidProviderRequest("Image part contains unsupported fields.")
                image = part.get("image_url")
                if not isinstance(image, Mapping) or set(image) - {"url"}:
                    raise InvalidProviderRequest("Image URL options are unsupported or ambiguous.")
                parts.append(self._convert_data_url(image.get("url")))
                continue
            if part_type == "input_audio":
                if set(part) - {"type", "input_audio"}:
                    raise InvalidProviderRequest("Audio part contains unsupported fields.")
                audio = part.get("input_audio")
                if not isinstance(audio, Mapping) or set(audio) != {"data", "format"}:
                    raise InvalidProviderRequest("Audio part is malformed.")
                fmt = audio.get("format")
                mime = {"wav": "audio/wav", "mp3": "audio/mpeg"}.get(fmt)
                data = audio.get("data")
                if mime is None or not isinstance(data, str):
                    raise InvalidProviderRequest("Audio format is unsupported.")
                self._validate_base64(data)
                parts.append({"inlineData": {"mimeType": mime, "data": data}})
                continue
            raise InvalidProviderRequest("Unsupported multimodal content part.")
        return parts

    def _convert_data_url(self, url: object) -> dict[str, object]:
        if not isinstance(url, str):
            raise InvalidProviderRequest("Image URL must be text.")
        match = _DATA_URL.fullmatch(url)
        if match is None:
            raise InvalidProviderRequest("Only bounded in-memory image data URLs are supported.")
        data = match.group("data")
        self._validate_base64(data)
        return {"inlineData": {"mimeType": match.group("mime"), "data": data}}

    def _validate_base64(self, data: str) -> None:
        try:
            decoded = base64.b64decode(data, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise InvalidProviderRequest("Inline media contains invalid base64.") from exc
        if len(decoded) > self._max_inline_bytes:
            raise InvalidProviderRequest("Inline media exceeded the configured bound.")

    def _match_pending(self, tool_calls: object) -> _PendingContinuation:
        if not isinstance(tool_calls, list) or not tool_calls:
            raise ContinuationMismatch("Assistant function calls are malformed.")
        first_id = tool_calls[0].get("id") if isinstance(tool_calls[0], Mapping) else None
        if not isinstance(first_id, str) or first_id not in self._pending_by_id:
            raise MissingContinuationState("Required continuation state is missing.")
        pending = self._pending_by_id[first_id]
        if len(tool_calls) != len(pending.calls):
            raise ContinuationMismatch("Function call count does not match continuation state.")
        for supplied, expected in zip(tool_calls, pending.calls):
            if not isinstance(supplied, Mapping):
                raise ContinuationMismatch("Assistant function call is malformed.")
            if set(supplied) - {"id", "type", "function"}:
                raise ContinuationMismatch("Assistant function call contains unsupported fields.")
            function = supplied.get("function")
            if (
                supplied.get("id") != expected.public_id
                or supplied.get("type") != "function"
                or not isinstance(function, Mapping)
                or set(function) - {"name", "arguments"}
                or function.get("name") != expected.name
            ):
                raise ContinuationMismatch("Function call identity does not match continuation state.")
            try:
                arguments = json.loads(function.get("arguments"))
            except (TypeError, json.JSONDecodeError):
                raise ContinuationMismatch("Function call arguments are malformed.") from None
            if arguments != expected.arguments:
                raise ContinuationMismatch("Function call arguments do not match continuation state.")
        return pending

    def _convert_tool_response(
        self, message: object, expected: _NativeCall
    ) -> dict[str, object]:
        if not isinstance(message, Mapping) or message.get("role") != "tool":
            raise ContinuationMismatch("Function responses must immediately follow their calls.")
        if set(message) - {"role", "content", "tool_call_id"}:
            raise ContinuationMismatch("Tool response contains unsupported fields.")
        if message.get("tool_call_id") != expected.public_id:
            raise ContinuationMismatch("Tool response ID does not match its function call.")
        content = message.get("content")
        if not isinstance(content, str):
            raise ContinuationMismatch("Tool response content must be text JSON or text.")
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            parsed = {"result": content}
        if not isinstance(parsed, dict):
            parsed = {"result": parsed}
        response: dict[str, object] = {"name": expected.name, "response": parsed}
        if expected.provider_id is not None:
            response["id"] = expected.provider_id
        return {"functionResponse": response}

    def _parse_response(self, payload: Mapping[str, object]) -> ProviderResult:
        candidates = payload.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != 1:
            raise MalformedProviderResponse("Gemini must return exactly one candidate.")
        candidate = candidates[0]
        if not isinstance(candidate, Mapping):
            raise MalformedProviderResponse("Gemini candidate is malformed.")
        content = candidate.get("content")
        if not isinstance(content, Mapping) or content.get("role") != "model":
            raise MalformedProviderResponse("Gemini model content is missing.")
        parts = content.get("parts")
        if not isinstance(parts, list) or not parts:
            raise MalformedProviderResponse("Gemini response parts are missing.")

        events: list[TextEvent | FunctionCallEvent] = []
        calls: list[_NativeCall] = []
        signature_seen = False
        for part in parts:
            if not isinstance(part, Mapping):
                raise MalformedProviderResponse("Gemini response part is malformed.")
            if "text" in part:
                if set(part) - {"text"}:
                    raise MalformedProviderResponse("Gemini text part contains unsupported fields.")
                text = part.get("text")
                if not isinstance(text, str):
                    raise MalformedProviderResponse("Gemini text part is malformed.")
                events.append(TextEvent(text=text))
                continue
            function_call = part.get("functionCall")
            if isinstance(function_call, Mapping):
                if set(part) - {"functionCall", "thoughtSignature"} or set(function_call) - {
                    "id",
                    "name",
                    "args",
                }:
                    raise MalformedProviderResponse("Gemini function call contains unsupported fields.")
                name = function_call.get("name")
                arguments = function_call.get("args")
                provider_id = function_call.get("id")
                signature = part.get("thoughtSignature")
                if (
                    not isinstance(name, str)
                    or not _SAFE_NAME.fullmatch(name)
                    or not isinstance(arguments, dict)
                    or (provider_id is not None and not isinstance(provider_id, str))
                ):
                    raise MalformedProviderResponse("Gemini function call is malformed.")
                if not calls:
                    if not isinstance(signature, str) or not signature:
                        raise MalformedProviderResponse("Gemini function call signature is missing.")
                    signature_seen = True
                elif signature is not None and not isinstance(signature, str):
                    raise MalformedProviderResponse("Gemini function call signature is malformed.")
                public_id = self._id_factory()
                native_call = _NativeCall(
                    public_id=public_id,
                    name=name,
                    arguments=deepcopy(arguments),
                    provider_id=provider_id,
                )
                calls.append(native_call)
                events.append(
                    FunctionCallEvent(
                        index=len(calls) - 1,
                        tool_call_id=public_id,
                        name=name,
                        arguments=deepcopy(arguments),
                    )
                )
                continue
            raise MalformedProviderResponse("Gemini returned an unsupported response part.")

        if calls:
            if not signature_seen:
                raise MalformedProviderResponse("Gemini function call signature is missing.")
            pending = _PendingContinuation(
                calls=tuple(calls),
                model_content=deepcopy(dict(content)),
                expires_at=self._clock() + self._pending_ttl_seconds,
            )
            if len(self._pending_by_id) + len(calls) > self._max_pending_calls:
                raise MalformedProviderResponse("Gemini continuation state capacity was exceeded.")
            public_ids = [call.public_id for call in calls]
            if len(set(public_ids)) != len(public_ids) or any(
                public_id in self._pending_by_id for public_id in public_ids
            ):
                raise MalformedProviderResponse("Generated continuation identity collided.")
            for call in calls:
                self._pending_by_id[call.public_id] = pending

        native_reason = candidate.get("finishReason", "UNSPECIFIED")
        if not isinstance(native_reason, str):
            raise MalformedProviderResponse("Gemini finish reason is malformed.")
        category = self._finish_category(native_reason, bool(calls))
        metadata = self._metadata(payload, len(candidates))
        finish = FinishEvent(category=category, native_reason=native_reason, metadata=metadata)
        return ProviderResult(events=tuple(events), finish=finish)

    @staticmethod
    def _finish_category(reason: str, has_calls: bool) -> FinishCategory:
        if has_calls:
            return FinishCategory.TOOL_CALLS
        return {
            "STOP": FinishCategory.STOP,
            "MAX_TOKENS": FinishCategory.LENGTH,
            "SAFETY": FinishCategory.SAFETY,
            "RECITATION": FinishCategory.SAFETY,
            "BLOCKLIST": FinishCategory.SAFETY,
            "PROHIBITED_CONTENT": FinishCategory.SAFETY,
        }.get(reason, FinishCategory.OTHER)

    @staticmethod
    def _metadata(payload: Mapping[str, object], candidate_count: int) -> ProviderMetadata:
        usage = payload.get("usageMetadata")
        if not isinstance(usage, Mapping):
            usage = {}

        def integer(name: str) -> int | None:
            value = usage.get(name)
            return value if isinstance(value, int) and value >= 0 else None

        return ProviderMetadata(
            provider="gemini",
            candidate_count=candidate_count,
            prompt_tokens=integer("promptTokenCount"),
            completion_tokens=integer("candidatesTokenCount"),
            total_tokens=integer("totalTokenCount"),
        )

    def _discard_pending(self, pending: _PendingContinuation) -> None:
        for call in pending.calls:
            self._pending_by_id.pop(call.public_id, None)
