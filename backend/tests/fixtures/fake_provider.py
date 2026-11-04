"""Deterministic, offline OpenAI-compatible provider replay fixtures.

The fixtures in this module are deliberately inert.  They model provider
payloads and failures without importing an SDK, opening a connection, or
implementing any production adapter behavior.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence, TypeAlias


JsonObject: TypeAlias = dict[str, Any]
StreamEvent: TypeAlias = JsonObject | str
DONE = "[DONE]"

MODEL = "offline-provider-model"
TEXT_COMPLETION_ID = "chatcmpl_text_fixture"
TOOL_COMPLETION_ID = "chatcmpl_tool_fixture"
TOOL_CALL_ID = "call_generate_image_fixture"
TOOL_NAME = "generate_image"
THOUGHT_SIGNATURE = "opaque-thought-signature-fixture"


def _chunk(
    completion_id: str,
    delta: Mapping[str, Any],
    finish_reason: str | None = None,
) -> JsonObject:
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": 1_700_000_000,
        "model": MODEL,
        "choices": [
            {
                "index": 0,
                "delta": dict(delta),
                "finish_reason": finish_reason,
            }
        ],
    }


TEXT_RESPONSE: JsonObject = {
    "id": TEXT_COMPLETION_ID,
    "object": "chat.completion",
    "created": 1_700_000_000,
    "model": MODEL,
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "The archive describes a stone courtyard.",
            },
            "finish_reason": "stop",
        }
    ],
}

TEXT_STREAM_EVENTS: tuple[StreamEvent, ...] = (
    _chunk(TEXT_COMPLETION_ID, {"role": "assistant", "content": ""}),
    _chunk(TEXT_COMPLETION_ID, {"content": "The archive describes "}),
    _chunk(TEXT_COMPLETION_ID, {"content": "a stone courtyard."}),
    _chunk(TEXT_COMPLETION_ID, {}, "stop"),
    DONE,
)

TOOL_ARGUMENT_FRAGMENTS = (
    '{"prompt":"A documented stone ',
    'courtyard","size":"1024x1024"}',
)
TOOL_ARGUMENTS = "".join(TOOL_ARGUMENT_FRAGMENTS)

TOOL_CALL: JsonObject = {
    "id": TOOL_CALL_ID,
    "type": "function",
    "function": {"name": TOOL_NAME, "arguments": TOOL_ARGUMENTS},
    "thought_signature": THOUGHT_SIGNATURE,
}

TOOL_RESPONSE: JsonObject = {
    "id": TOOL_COMPLETION_ID,
    "object": "chat.completion",
    "created": 1_700_000_000,
    "model": MODEL,
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [TOOL_CALL],
            },
            "finish_reason": "tool_calls",
        }
    ],
}


def _tool_delta(arguments: str) -> JsonObject:
    """Return one independently replayable incremental tool-call delta."""

    return {
        "tool_calls": [
            {
                "index": 0,
                "id": TOOL_CALL_ID,
                "type": "function",
                "function": {"name": TOOL_NAME, "arguments": arguments},
                "thought_signature": THOUGHT_SIGNATURE,
            }
        ]
    }


TOOL_STREAM_EVENTS: tuple[StreamEvent, ...] = (
    _chunk(TOOL_COMPLETION_ID, {"role": "assistant", "content": None}),
    _chunk(TOOL_COMPLETION_ID, _tool_delta(TOOL_ARGUMENT_FRAGMENTS[0])),
    _chunk(TOOL_COMPLETION_ID, _tool_delta(TOOL_ARGUMENT_FRAGMENTS[1])),
    _chunk(TOOL_COMPLETION_ID, {}, "tool_calls"),
    DONE,
)

TOOL_RESULT: JsonObject = {
    "role": "tool",
    "tool_call_id": TOOL_CALL_ID,
    "content": '{"image_id":"image-fixture-1","status":"stored"}',
}

TOOL_CONTINUATION_MESSAGES: tuple[JsonObject, ...] = (
    {"role": "user", "content": "Reconstruct the documented courtyard."},
    {"role": "assistant", "content": None, "tool_calls": [TOOL_CALL]},
    TOOL_RESULT,
)

CONTINUATION_RESPONSE: JsonObject = {
    "id": "chatcmpl_continuation_fixture",
    "object": "chat.completion",
    "created": 1_700_000_000,
    "model": MODEL,
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "The grounded reconstruction is ready.",
            },
            "finish_reason": "stop",
        }
    ],
}

CONTINUATION_STREAM_EVENTS: tuple[StreamEvent, ...] = (
    _chunk("chatcmpl_continuation_fixture", {"role": "assistant", "content": ""}),
    _chunk("chatcmpl_continuation_fixture", {"content": "The grounded reconstruction "}),
    _chunk("chatcmpl_continuation_fixture", {"content": "is ready."}),
    _chunk("chatcmpl_continuation_fixture", {}, "stop"),
    DONE,
)

# Provider incompatibility: the delta has an ID and name but no required index.
MISSING_INDEX_STREAM_EVENTS: tuple[StreamEvent, ...] = (
    _chunk(
        "chatcmpl_missing_index_fixture",
        {
            "tool_calls": [
                {
                    "id": TOOL_CALL_ID,
                    "type": "function",
                    "function": {"name": TOOL_NAME, "arguments": '{"prompt":'},
                }
            ]
        },
    ),
    _chunk("chatcmpl_missing_index_fixture", {}, "tool_calls"),
    DONE,
)

# The argument fragments are incremental but concatenate to malformed JSON.
INVALID_JSON_ARGUMENT_FRAGMENTS = ('{"prompt":"courtyard"', ',"size":}')
INVALID_JSON_STREAM_EVENTS: tuple[StreamEvent, ...] = (
    _chunk("chatcmpl_invalid_json_fixture", _tool_delta(INVALID_JSON_ARGUMENT_FRAGMENTS[0])),
    _chunk("chatcmpl_invalid_json_fixture", _tool_delta(INVALID_JSON_ARGUMENT_FRAGMENTS[1])),
    _chunk("chatcmpl_invalid_json_fixture", {}, "tool_calls"),
    DONE,
)

MISMATCHED_TOOL_CONTINUATION_MESSAGES: tuple[JsonObject, ...] = (
    {"role": "assistant", "content": None, "tool_calls": [TOOL_CALL]},
    {
        "role": "tool",
        "tool_call_id": "call_different_fixture",
        "content": '{"status":"stored"}',
    },
)


@dataclass(frozen=True)
class FailureFixture:
    """An inert description of a provider failure to replay."""

    name: str
    status_code: int
    error_type: str
    code: str
    message: str


TIMEOUT_FAILURE = FailureFixture(
    name="timeout",
    status_code=504,
    error_type="provider_timeout",
    code="PROVIDER_TIMEOUT",
    message="Offline provider fixture timed out.",
)
RATE_LIMIT_FAILURE = FailureFixture(
    name="rate_limit",
    status_code=429,
    error_type="rate_limit_error",
    code="RATE_LIMITED",
    message="Offline provider fixture was throttled.",
)
SERVER_ERROR_FAILURE = FailureFixture(
    name="server_error",
    status_code=503,
    error_type="provider_error",
    code="PROVIDER_UNAVAILABLE",
    message="Offline provider fixture returned a 5xx response.",
)

FAILURE_FIXTURES = {
    fixture.name: fixture
    for fixture in (TIMEOUT_FAILURE, RATE_LIMIT_FAILURE, SERVER_ERROR_FAILURE)
}


class FakeProviderFailure(RuntimeError):
    """Raised only when a test explicitly replays a failure fixture."""

    def __init__(self, fixture: FailureFixture):
        super().__init__(fixture.message)
        self.fixture = fixture


class FakeProvider:
    """Small replay fake with no network or credential dependencies."""

    def __init__(
        self,
        response: Mapping[str, Any],
        stream_events: Sequence[StreamEvent],
        failure: FailureFixture | None = None,
    ) -> None:
        self._response = deepcopy(dict(response))
        self._stream_events = deepcopy(tuple(stream_events))
        self._failure = failure

    def complete(self) -> JsonObject:
        if self._failure is not None:
            raise FakeProviderFailure(self._failure)
        return deepcopy(self._response)

    def stream(self) -> Iterator[StreamEvent]:
        if self._failure is not None:
            raise FakeProviderFailure(self._failure)
        yield from deepcopy(self._stream_events)


TEXT_PROVIDER = FakeProvider(TEXT_RESPONSE, TEXT_STREAM_EVENTS)
TOOL_PROVIDER = FakeProvider(TOOL_RESPONSE, TOOL_STREAM_EVENTS)
CONTINUATION_PROVIDER = FakeProvider(
    CONTINUATION_RESPONSE,
    CONTINUATION_STREAM_EVENTS,
)


def replay_stream(events: Sequence[StreamEvent]) -> Iterator[StreamEvent]:
    """Replay arbitrary fixture events without exposing shared mutable state."""

    yield from deepcopy(tuple(events))
