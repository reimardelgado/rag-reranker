from copy import deepcopy
from datetime import datetime, timezone
import json

import pytest
from pydantic import ValidationError

from backend.api.chat import (
    ImageToolContinuation,
    ImageToolContinuationLimits,
    ImageToolEnforcer,
    InvalidImageToolCall,
)
from backend.chat_contracts import ChatCompletionRequest
from backend.grounding.evidence import build_evidence_packet
from backend.grounding.instruction import GroundedImageInstruction
from backend.grounding.provenance import (
    ADJACENT_LABEL,
    DOCUMENTARY_DISCLAIMER,
    ImageProvenance,
    InvalidProvenanceError,
    build_image_provenance,
    render_adjacent_provenance,
)
from backend.grounding.validation import validate_grounded_instruction
from backend.providers.base import (
    FinishCategory,
    FinishEvent,
    FunctionCallEvent,
    OpenAIResponseNormalizer,
    ProviderMetadata,
    TextEvent,
)


METADATA = ProviderMetadata(
    provider="PRIVATE_PROVIDER_MARKER",
    candidate_count=1,
)


async def _events(*events):
    for event in events:
        yield event


def _finish(category=FinishCategory.STOP):
    return FinishEvent(category, "PRIVATE_NATIVE_SIGNATURE", METADATA)


def _grounding(*, injected_source="Survey.pdf"):
    query = "Generate an image of the documented courtyard"
    packet = build_evidence_packet(
        query,
        [
            {
                "reference": {
                    "reference_id": "ref-a",
                    "source": injected_source,
                    "page": 12,
                },
                "content": "RAW_EVIDENCE_SECRET auth=do-not-disclose",
            },
            {
                "reference": {
                    "reference_id": "ref-b",
                    "source": "Excavation.pdf",
                    "page": 8,
                },
                "content": "The entrance faced south.",
            },
        ],
    )
    instruction = GroundedImageInstruction.model_validate(
        {
            "version": "grounded-image-v1",
            "query": query,
            "documented_facts": [
                {
                    "text": "The courtyard used large stone blocks.",
                    "reference_ids": ["ref-a"],
                }
            ],
            "inferred_elements": ["Roof material is interpretive."],
            "uncertainties": ["Building height is unknown."],
            "forbidden_additions": ["Modern vehicles"],
            "evidence_refs": [item.model_dump() for item in packet.references],
        }
    )
    return validate_grounded_instruction(
        instruction,
        packet,
        size="1024x1024",
        nonce="continuation_nonce_1",
    )


def _provenance(grounding=None):
    return build_image_provenance(
        grounding or _grounding(),
        retrieval_model_id="retrieval-public-v1",
        chat_model_id="chat-public-v1",
        image_model_id="image-public-v1",
        generated_at=datetime(2026, 10, 30, 12, 34, 56, tzinfo=timezone.utc),
    )


def _request(messages=None):
    return ChatCompletionRequest.model_validate(
        {
            "model": "chat-public-v1",
            "messages": messages
            or [{"role": "user", "content": "Generate a reconstruction"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "generate_image",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "prompt": {"type": "string"},
                                "size": {"type": "string"},
                            },
                            "required": ["prompt"],
                            "additionalProperties": False,
                        },
                    },
                }
            ],
        }
    )


def _decode_sse(chunks):
    decoded = []
    for chunk in chunks:
        value = chunk.decode().strip()[6:]
        decoded.append(value if value == "[DONE]" else json.loads(value))
    return decoded


async def _normalize(events, *, stream):
    normalizer = OpenAIResponseNormalizer(argument_fragment_chars=17)
    if not stream:
        return await normalizer.non_stream(
            events,
            model="chat-public-v1",
            response_id="chatcmpl-test",
            created=1,
        )
    chunks = [
        chunk
        async for chunk in normalizer.stream(
            events,
            model="chat-public-v1",
            response_id="chatcmpl-test",
            created=1,
        )
    ]
    return _decode_sse(chunks)


def _assistant_from_output(output, *, stream):
    if not stream:
        return output["choices"][0]["message"]
    tool_deltas = []
    for payload in output:
        if payload == "[DONE]":
            continue
        delta = payload["choices"][0]["delta"]
        if "tool_calls" in delta:
            tool_deltas.append(delta["tool_calls"][0])
    first = tool_deltas[0]
    arguments = "".join(item["function"].get("arguments", "") for item in tool_deltas)
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": first["id"],
                "type": "function",
                "function": {"name": first["function"]["name"], "arguments": arguments},
            }
        ],
    }


def _content_from_output(output, *, stream):
    if not stream:
        return output["choices"][0]["message"]["content"]
    return "".join(
        payload["choices"][0]["delta"].get("content", "")
        for payload in output
        if payload != "[DONE]"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_two_round_tool_continuation_appends_adjacent_provenance_in_both_modes(stream):
    grounding = _grounding()
    request = _request()
    original = request.model_dump(exclude_unset=True)
    enforcer = ImageToolEnforcer(request, grounding)
    forced = enforcer.forced_request(request)

    first = await _normalize(
        enforcer.enforce(
            _events(
                FunctionCallEvent(
                    0,
                    "call_public_image",
                    "generate_image",
                    {"prompt": "provider must not control this", "size": "1024x1024"},
                ),
                _finish(FinishCategory.TOOL_CALLS),
            )
        ),
        stream=stream,
    )
    assistant = _assistant_from_output(first, stream=stream)
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"])["prompt"] == grounding.canonical_json

    continuation_payload = original | {
        "messages": [
            *original["messages"],
            assistant,
            {
                "role": "tool",
                "tool_call_id": "call_public_image",
                "content": '{"image_id":"PRIVATE_IMAGE_RESULT","status":"stored"}',
            },
        ],
        "stream": stream,
    }
    continuation_request = ChatCompletionRequest.model_validate(continuation_payload)
    continuation_before = continuation_request.model_dump(exclude_unset=True)
    boundary = ImageToolContinuation("call_public_image", _provenance(grounding))
    detached = boundary.continuation_request(continuation_request)
    final = await _normalize(
        boundary.append_provenance(
            _events(TextEvent("The grounded reconstruction is ready."), _finish())
        ),
        stream=stream,
    )
    content = _content_from_output(final, stream=stream)

    assert detached is not continuation_request
    assert continuation_request.model_dump(exclude_unset=True) == continuation_before
    assert request.model_dump(exclude_unset=True) == original
    assert forced.tool_choice.function.name == "generate_image"
    assert content.index("The grounded reconstruction is ready.") < content.index(ADJACENT_LABEL)
    assert DOCUMENTARY_DISCLAIMER in content
    assert "<details><summary>Provenance</summary>" in content
    assert '"prompt_hash":"' in content
    assert '"reference_id":"ref-a"' in content
    assert '"documented_elements"' in content
    assert '"inferred_elements"' in content
    assert '"uncertainties"' in content
    assert '"forbidden_items"' in content
    assert '"timestamp":"2026-10-30T12:34:56Z"' in content
    for private in (
        "RAW_EVIDENCE_SECRET",
        "do-not-disclose",
        "PRIVATE_IMAGE_RESULT",
        "PRIVATE_PROVIDER_MARKER",
        "PRIVATE_NATIVE_SIGNATURE",
        "provider must not control this",
    ):
        assert private not in content


def test_provenance_is_derived_from_validated_prompt_and_escapes_display_injection():
    injected = "Archive </details> ``` <script>private()</script>"
    grounding = _grounding(injected_source=injected)
    provenance = _provenance(grounding)
    rendered = render_adjacent_provenance(provenance)
    sidecar = rendered.split("```json\n", 1)[1].split("\n```", 1)[0]

    assert provenance.prompt_hash == grounding.prompt_hash
    assert provenance.prompt_version == "grounded-image-v1"
    assert json.loads(sidecar)["references"][0]["source"] == injected
    assert rendered.count("</details>") == 1
    assert "<script>" not in rendered
    assert "``` <script>" not in rendered
    assert "RAW_EVIDENCE_SECRET" not in rendered
    with pytest.raises(ValidationError):
        ImageProvenance.model_validate(provenance.model_dump() | {"secret": "forbidden"})


def test_provenance_rejects_tampering_naive_time_and_output_overflow():
    grounding = _grounding()
    tampered = grounding.model_copy(update={"prompt_hash": "0" * 64})

    with pytest.raises(InvalidProvenanceError, match="hash"):
        _provenance(tampered)
    with pytest.raises(InvalidProvenanceError, match="timezone"):
        build_image_provenance(
            grounding,
            retrieval_model_id="retrieval",
            chat_model_id="chat",
            image_model_id="image",
            generated_at=datetime(2026, 10, 30),
        )
    with pytest.raises(InvalidProvenanceError, match="configured bound"):
        render_adjacent_provenance(_provenance(grounding), max_sidecar_chars=8)


@pytest.mark.parametrize(
    "messages",
    [
        [{"role": "user", "content": "missing pair"}],
        [
            {"role": "assistant", "content": None, "tool_calls": []},
            {"role": "tool", "tool_call_id": "call_public_image", "content": "{}"},
        ],
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_other",
                        "type": "function",
                        "function": {"name": "generate_image", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_other", "content": "{}"},
        ],
        [
            {"role": "tool", "tool_call_id": "call_public_image", "content": "{}"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_public_image",
                        "type": "function",
                        "function": {"name": "generate_image", "arguments": "{}"},
                    }
                ],
            },
        ],
    ],
)
def test_continuation_rejects_incomplete_mismatched_or_reordered_public_pairs(messages):
    boundary = ImageToolContinuation("call_public_image", _provenance())
    with pytest.raises(InvalidImageToolCall):
        boundary.continuation_request(_request(messages=messages))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "events",
    [
        (FunctionCallEvent(0, "call_nested", "generate_image", {"prompt": "again"}), _finish(FinishCategory.TOOL_CALLS)),
        (TextEvent("partial"), _finish(FinishCategory.TOOL_CALLS)),
        (TextEvent("partial"), _finish(), _finish()),
        (_finish(),),
    ],
)
async def test_final_turn_rejects_recursive_calls_bad_finishes_and_empty_text_without_output(events):
    boundary = ImageToolContinuation("call_public_image", _provenance())
    released = []
    with pytest.raises(InvalidImageToolCall):
        async for event in boundary.append_provenance(_events(*events)):
            released.append(event)
    assert released == []


@pytest.mark.asyncio
async def test_continuation_bounds_fail_closed_before_any_output():
    event_limited = ImageToolContinuation(
        "call_public_image",
        _provenance(),
        limits=ImageToolContinuationLimits(max_buffered_events=1),
    )
    text_limited = ImageToolContinuation(
        "call_public_image",
        _provenance(),
        limits=ImageToolContinuationLimits(max_text_chars=3),
    )

    for boundary in (event_limited, text_limited):
        released = []
        with pytest.raises(InvalidImageToolCall):
            async for event in boundary.append_provenance(
                _events(TextEvent("bounded text"), _finish())
            ):
                released.append(event)
        assert released == []
