import json

import pytest

from backend.tests.fixtures.fake_provider import (
    CONTINUATION_PROVIDER,
    DONE,
    FAILURE_FIXTURES,
    INVALID_JSON_ARGUMENT_FRAGMENTS,
    INVALID_JSON_STREAM_EVENTS,
    MISMATCHED_TOOL_CONTINUATION_MESSAGES,
    MISSING_INDEX_STREAM_EVENTS,
    RATE_LIMIT_FAILURE,
    SERVER_ERROR_FAILURE,
    TEXT_PROVIDER,
    THOUGHT_SIGNATURE,
    TIMEOUT_FAILURE,
    TOOL_ARGUMENTS,
    TOOL_CALL_ID,
    TOOL_CONTINUATION_MESSAGES,
    TOOL_NAME,
    TOOL_PROVIDER,
    FakeProvider,
    FakeProviderFailure,
    replay_stream,
)


def _choice(event):
    return event["choices"][0]


def _tool_deltas(events):
    return [
        _choice(event)["delta"]["tool_calls"][0]
        for event in events
        if event != DONE and _choice(event)["delta"].get("tool_calls")
    ]


def test_text_replay_matches_non_stream_content_and_done_ordering():
    response = TEXT_PROVIDER.complete()
    events = list(TEXT_PROVIDER.stream())

    content = "".join(
        _choice(event)["delta"].get("content", "")
        for event in events
        if event != DONE
    )

    assert content == response["choices"][0]["message"]["content"]
    assert _choice(events[-2])["finish_reason"] == "stop"
    assert events[-1] == DONE


def test_tool_replay_has_stable_identity_incremental_arguments_and_terminal_done():
    response = TOOL_PROVIDER.complete()
    events = list(TOOL_PROVIDER.stream())
    deltas = _tool_deltas(events)

    assert len(deltas) == 2
    assert {delta["index"] for delta in deltas} == {0}
    assert {delta["id"] for delta in deltas} == {TOOL_CALL_ID}
    assert {delta["function"]["name"] for delta in deltas} == {TOOL_NAME}
    assert {delta["thought_signature"] for delta in deltas} == {THOUGHT_SIGNATURE}
    assert "".join(delta["function"]["arguments"] for delta in deltas) == TOOL_ARGUMENTS
    assert json.loads(TOOL_ARGUMENTS)["size"] == "1024x1024"
    assert response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] == TOOL_ARGUMENTS
    assert _choice(events[-2])["finish_reason"] == "tool_calls"
    assert events[-1] == DONE


def test_replays_are_deterministic_and_do_not_share_mutable_payloads():
    first_response = TOOL_PROVIDER.complete()
    first_events = list(TOOL_PROVIDER.stream())
    first_response["choices"][0]["message"]["tool_calls"][0]["id"] = "mutated"
    first_events[1]["choices"][0]["delta"]["tool_calls"][0]["id"] = "mutated"

    assert TOOL_PROVIDER.complete()["choices"][0]["message"]["tool_calls"][0]["id"] == TOOL_CALL_ID
    assert _tool_deltas(list(TOOL_PROVIDER.stream()))[0]["id"] == TOOL_CALL_ID
    assert list(replay_stream(TOOL_PROVIDER.stream())) == list(TOOL_PROVIDER.stream())


def test_continuation_links_assistant_call_to_tool_result_and_final_text():
    assistant = TOOL_CONTINUATION_MESSAGES[-2]
    tool_result = TOOL_CONTINUATION_MESSAGES[-1]
    call = assistant["tool_calls"][0]

    assert assistant["role"] == "assistant"
    assert call["id"] == TOOL_CALL_ID
    assert tool_result["role"] == "tool"
    assert tool_result["tool_call_id"] == call["id"]
    assert CONTINUATION_PROVIDER.complete()["choices"][0]["message"] == {
        "role": "assistant",
        "content": "The grounded reconstruction is ready.",
    }
    assert list(CONTINUATION_PROVIDER.stream())[-1] == DONE


def test_malformed_and_provider_specific_variants_are_explicit():
    missing_index_call = _tool_deltas(MISSING_INDEX_STREAM_EVENTS)[0]
    invalid_json_calls = _tool_deltas(INVALID_JSON_STREAM_EVENTS)
    invalid_arguments = "".join(
        call["function"]["arguments"] for call in invalid_json_calls
    )

    assert "index" not in missing_index_call
    assert tuple(call["function"]["arguments"] for call in invalid_json_calls) == INVALID_JSON_ARGUMENT_FRAGMENTS
    with pytest.raises(json.JSONDecodeError):
        json.loads(invalid_arguments)

    mismatched_assistant, mismatched_result = MISMATCHED_TOOL_CONTINUATION_MESSAGES
    assert mismatched_result["tool_call_id"] != mismatched_assistant["tool_calls"][0]["id"]
    assert TOOL_PROVIDER.complete()["choices"][0]["message"]["tool_calls"][0]["thought_signature"] == THOUGHT_SIGNATURE


@pytest.mark.parametrize(
    ("fixture", "status_code"),
    [
        (TIMEOUT_FAILURE, 504),
        (RATE_LIMIT_FAILURE, 429),
        (SERVER_ERROR_FAILURE, 503),
    ],
)
def test_failure_fixtures_replay_without_external_calls(fixture, status_code):
    provider = FakeProvider({}, (), failure=fixture)

    with pytest.raises(FakeProviderFailure) as completion_error:
        provider.complete()
    with pytest.raises(FakeProviderFailure) as stream_error:
        list(provider.stream())

    assert completion_error.value.fixture is fixture
    assert stream_error.value.fixture is fixture
    assert fixture.status_code == status_code
    assert FAILURE_FIXTURES[fixture.name] is fixture
