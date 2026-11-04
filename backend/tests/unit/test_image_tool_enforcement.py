from copy import deepcopy

import pytest

from backend.api.chat import (
    ImageToolEnforcementLimits,
    ImageToolEnforcer,
    InvalidImageToolCall,
    UnsupportedImageTool,
)
from backend.chat_contracts import ChatCompletionRequest
from backend.grounding.evidence import build_evidence_packet
from backend.grounding.instruction import GroundedImageInstruction
from backend.grounding.validation import validate_grounded_instruction
from backend.providers.base import (
    FinishCategory,
    FinishEvent,
    FunctionCallEvent,
    ProviderMetadata,
    TextEvent,
)


METADATA = ProviderMetadata(provider="offline", candidate_count=1)


def _grounding(*, size="1024x1024", quality="standard", style="archival"):
    packet = build_evidence_packet(
        "Generate an image of the documented site",
        [
            {
                "reference": {"reference_id": "ref-a", "source": "A.pdf", "page": 1},
                "content": "The entrance faced south.",
            },
            {
                "reference": {"reference_id": "ref-b", "source": "B.pdf", "page": 2},
                "content": "The wall used large stone blocks.",
            },
        ],
    )
    instruction = GroundedImageInstruction.model_validate(
        {
            "version": "grounded-image-v1",
            "query": packet.query,
            "documented_facts": [
                {"text": "The entrance faced south.", "reference_ids": ["ref-a"]},
                {"text": "The wall used large stone blocks.", "reference_ids": ["ref-b"]},
            ],
            "forbidden_additions": ["Modern vehicles"],
            "evidence_refs": [reference.model_dump() for reference in packet.references],
        }
    )
    return validate_grounded_instruction(
        instruction,
        packet,
        size=size,
        quality=quality,
        style=style,
        nonce="offline_nonce_1234",
    )


def _request(*, tools=None, tool_choice="auto"):
    if tools is None:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "generate_image",
                    "description": "Generate one image",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "prompt": {"type": "string", "minLength": 1},
                            "size": {"type": "string", "enum": ["1024x1024"]},
                            "quality": {"type": "string", "enum": ["standard", "hd"]},
                            "style": {"type": "string", "enum": ["archival"]},
                        },
                        "required": ["prompt"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "unrelated_tool",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ]
    return ChatCompletionRequest.model_validate(
        {
            "model": "offline-model",
            "messages": [{"role": "user", "content": "Generate the site"}],
            "tools": tools,
            "tool_choice": tool_choice,
            "client_extension": {"preserved": True},
        }
    )


def _finish(category=FinishCategory.TOOL_CALLS):
    return FinishEvent(category=category, native_reason="offline", metadata=METADATA)


async def _events(*events, consumed=None):
    for event in events:
        if consumed is not None:
            consumed.append(type(event).__name__)
        yield event


def test_forced_request_requires_advertisement_and_is_detached_without_field_loss():
    request = _request(tool_choice="none")
    original = request.model_dump(exclude_unset=True)
    enforcer = ImageToolEnforcer(request, _grounding())

    forced = enforcer.forced_request(request)

    assert forced.tool_choice.model_dump() == {
        "type": "function",
        "function": {"name": "generate_image"},
    }
    assert forced.model_dump(exclude_unset=True)["client_extension"] == {"preserved": True}
    assert forced.model_dump(exclude_unset=True)["tools"] == original["tools"]
    assert request.model_dump(exclude_unset=True) == original


@pytest.mark.parametrize(
    "tools",
    [
        [],
        [
            {
                "type": "function",
                "function": {
                    "name": "other",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        [
            {
                "type": "function",
                "function": {
                    "name": "generate_image",
                    "parameters": {"type": "array", "properties": {}},
                },
            }
        ],
        [
            {
                "type": "function",
                "function": {
                    "name": "generate_image",
                    "parameters": {
                        "type": "object",
                        "properties": {"prompt": {"type": "integer"}},
                    },
                },
            }
        ],
        [
            {
                "type": "function",
                "function": {
                    "name": "generate_image",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "prompt": {"type": "string"},
                            "seed": {"type": "integer"},
                        },
                        "required": ["prompt", "seed"],
                    },
                },
            }
        ],
    ],
)
def test_missing_or_incompatible_image_advertisement_fails_closed(tools):
    with pytest.raises(UnsupportedImageTool):
        ImageToolEnforcer(_request(tools=tools), _grounding())


def test_duplicate_image_advertisements_fail_closed():
    tools = _request().model_dump(exclude_unset=True)["tools"]
    tools.append(deepcopy(tools[0]))

    with pytest.raises(UnsupportedImageTool):
        ImageToolEnforcer(_request(tools=tools), _grounding())


@pytest.mark.asyncio
async def test_complete_call_is_buffered_sanitized_and_does_not_mutate_inputs():
    grounding = _grounding()
    enforcer = ImageToolEnforcer(_request(), grounding)
    arguments = {
        "prompt": "provider supplied and must never escape",
        "size": "1024x1024",
        "quality": "hd",
        "style": "archival",
    }
    original = deepcopy(arguments)
    call = FunctionCallEvent(0, "call_offline", "generate_image", arguments)
    consumed = []
    enforced = enforcer.enforce(_events(call, _finish(), consumed=consumed))

    first = await anext(enforced)

    assert consumed == ["FunctionCallEvent", "FinishEvent"]
    assert first.arguments == {
        "prompt": grounding.image_spec.canonical_prompt,
        "size": "1024x1024",
        "quality": "standard",
        "style": "archival",
    }
    assert first is not call
    assert arguments == original
    assert await anext(enforced) == _finish()
    with pytest.raises(StopAsyncIteration):
        await anext(enforced)


@pytest.mark.parametrize(
    ("events", "match"),
    [
        ((FunctionCallEvent(0, "call", "other", {"prompt": "x"}), _finish()), "unexpected"),
        ((FunctionCallEvent(0, "a", "generate_image", {"prompt": "x"}), FunctionCallEvent(1, "b", "generate_image", {"prompt": "x"}), _finish()), "sequence"),
        ((TextEvent("unconstrained"), FunctionCallEvent(0, "call", "generate_image", {"prompt": "x"}), _finish()), "assistant text"),
        ((FunctionCallEvent(0, "call", "generate_image", {"prompt": "x"}), _finish(FinishCategory.STOP)), "finish reason"),
        ((FunctionCallEvent(0, "call", "generate_image", {"prompt": "x"}),), "sequence"),
        ((_finish(), FunctionCallEvent(0, "call", "generate_image", {"prompt": "x"})), "sequence"),
    ],
)
@pytest.mark.asyncio
async def test_invalid_provider_sequences_release_nothing(events, match):
    enforcer = ImageToolEnforcer(_request(), _grounding())

    with pytest.raises(InvalidImageToolCall, match=match):
        await anext(enforcer.enforce(_events(*events)))


@pytest.mark.parametrize(
    ("arguments", "match"),
    [
        ({}, "require a prompt"),
        ({"prompt": 7}, "must be text"),
        ({"prompt": "x", "seed": 3}, "unsupported fields"),
        ({"prompt": "x", "size": "512x512"}, "violates its schema"),
        ({"prompt": "x", "quality": object()}, "malformed"),
    ],
)
@pytest.mark.asyncio
async def test_malformed_unknown_or_schema_invalid_arguments_release_nothing(arguments, match):
    enforcer = ImageToolEnforcer(_request(), _grounding())
    call = FunctionCallEvent(0, "call", "generate_image", arguments)

    with pytest.raises(InvalidImageToolCall, match=match):
        await anext(enforcer.enforce(_events(call, _finish())))


@pytest.mark.asyncio
async def test_undeclared_options_argument_and_configured_bounds_fail_closed():
    tool = _request().model_dump(exclude_unset=True)["tools"][0]
    del tool["function"]["parameters"]["properties"]["size"]
    request = _request(tools=[tool])
    undeclared = ImageToolEnforcer(request, _grounding(size=None))
    call = FunctionCallEvent(
        0, "call", "generate_image", {"prompt": "x", "size": "1024x1024"}
    )
    with pytest.raises(InvalidImageToolCall, match="unsupported fields"):
        await anext(undeclared.enforce(_events(call, _finish())))

    bounded = ImageToolEnforcer(
        _request(),
        _grounding(),
        limits=ImageToolEnforcementLimits(max_buffered_events=1),
    )
    with pytest.raises(InvalidImageToolCall, match="event bound"):
        await anext(
            bounded.enforce(
                _events(
                    FunctionCallEvent(0, "call", "generate_image", {"prompt": "x"}),
                    _finish(),
                )
            )
        )

    tiny = ImageToolEnforcer(
        _request(),
        _grounding(),
        limits=ImageToolEnforcementLimits(max_arguments_chars=8),
    )
    with pytest.raises(InvalidImageToolCall, match="configured bound"):
        await anext(
            tiny.enforce(
                _events(
                    FunctionCallEvent(0, "call", "generate_image", {"prompt": "long"}),
                    _finish(),
                )
            )
        )


def test_enforcement_limits_must_be_positive():
    with pytest.raises(ValueError, match="must be positive"):
        ImageToolEnforcementLimits(max_buffered_events=0)
