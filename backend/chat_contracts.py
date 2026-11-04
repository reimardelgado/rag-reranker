"""OpenAI-compatible chat request contracts and retrieval text helpers.

These models deliberately preserve fields that this backend does not execute
yet.  The API boundary can therefore accept current OpenAI chat payloads
without silently discarding tool, multimodal, or provider-specific metadata.
"""

from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field


class ForwardCompatibleModel(BaseModel):
    """Base contract that retains supported fields added by clients/providers."""

    model_config = ConfigDict(extra="allow")


class ChatContentPart(ForwardCompatibleModel):
    """One multimodal content part, including unknown future part fields."""

    type: str
    text: str | None = None


ChatContent: TypeAlias = str | list[ChatContentPart]


class ChatFunctionCall(ForwardCompatibleModel):
    name: str
    arguments: str


class AssistantToolCall(ForwardCompatibleModel):
    id: str
    type: str = "function"
    function: ChatFunctionCall


class SystemMessage(ForwardCompatibleModel):
    role: Literal["system"]
    content: ChatContent


class UserMessage(ForwardCompatibleModel):
    role: Literal["user"]
    content: ChatContent


class AssistantMessage(ForwardCompatibleModel):
    role: Literal["assistant"]
    content: ChatContent | None = None
    tool_calls: list[AssistantToolCall] | None = None


class ToolMessage(ForwardCompatibleModel):
    role: Literal["tool"]
    content: ChatContent
    tool_call_id: str


ChatMessage: TypeAlias = Annotated[
    SystemMessage | UserMessage | AssistantMessage | ToolMessage,
    Field(discriminator="role"),
]


class ChatFunctionDefinition(ForwardCompatibleModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None
    strict: bool | None = None


class ChatTool(ForwardCompatibleModel):
    type: str
    function: ChatFunctionDefinition | None = None


class ToolChoiceFunction(ForwardCompatibleModel):
    name: str


class ToolChoiceObject(ForwardCompatibleModel):
    type: str
    function: ToolChoiceFunction | None = None


ToolChoice: TypeAlias = str | ToolChoiceObject


class StreamOptions(ForwardCompatibleModel):
    include_usage: bool | None = None


class ChatCompletionRequest(ForwardCompatibleModel):
    """Forward-compatible request envelope for ``/v1/chat/completions``."""

    model: str
    messages: list[ChatMessage]
    temperature: float = 0.6
    stream: bool = False
    stream_options: StreamOptions | None = None
    tools: list[ChatTool] | None = None
    tool_choice: ToolChoice | None = None
    top_k: int = Field(default=6, ge=1, le=50)


# Compatibility name retained for callers that imported the original request
# model from backend.api.chat or backend.app.
OpenAIChatRequest = ChatCompletionRequest


def _message_value(message: ChatMessage | Mapping[str, Any], field: str) -> Any:
    if isinstance(message, BaseModel):
        return getattr(message, field, None)
    return message.get(field)


def textual_content_parts(content: Any) -> tuple[str, ...]:
    """Return text from a string or standard ``type=text`` content parts.

    Non-text attachments are intentionally ignored.  The source objects are
    only inspected, never normalized or mutated.
    """

    if isinstance(content, str):
        return (content,) if content else ()
    if not isinstance(content, Sequence) or isinstance(content, (bytes, bytearray)):
        return ()

    text_parts: list[str] = []
    for part in content:
        if isinstance(part, BaseModel):
            part_type = getattr(part, "type", None)
            text = getattr(part, "text", None)
        elif isinstance(part, Mapping):
            part_type = part.get("type")
            text = part.get("text")
        else:
            continue
        if part_type == "text" and isinstance(text, str) and text:
            text_parts.append(text)
    return tuple(text_parts)


def extract_user_query(
    messages: Sequence[ChatMessage | Mapping[str, Any]],
) -> str:
    """Derive retrieval text from every text part of the latest user turn.

    Looking only at the latest user turn preserves the endpoint's established
    conversational behavior.  Crucially, an attachment-only latest turn
    returns an empty query rather than falling back to stale earlier text.
    """

    for message in reversed(messages):
        if _message_value(message, "role") == "user":
            return "\n".join(
                textual_content_parts(_message_value(message, "content"))
            )
    return ""
