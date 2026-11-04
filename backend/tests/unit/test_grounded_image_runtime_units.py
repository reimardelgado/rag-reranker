import json

import httpx
import pytest

from backend.chat_contracts import ChatCompletionRequest
from backend.grounding.retrieval_evidence import (
    evidence_packet_from_blocks,
    parse_retrieval_block,
    reference_id_for,
    retrieval_query_for_image,
)
from backend.providers.base import ContinuationMismatch, MissingContinuationState
from backend.providers.gemini_native import GeminiNativeAdapter


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Genera una imagen del oppidum ibero", "oppidum ibero"),
        ("Create an image of the Iberian sanctuary", "Iberian sanctuary"),
        ("Dibuja una reconstrucción de la muralla de Puente Tablas", "muralla de Puente Tablas"),
        ("genera imagen", "genera imagen"),
    ],
)
def test_retrieval_query_strips_only_leading_request_framing(query, expected):
    assert retrieval_query_for_image(query) == expected


def test_parse_block_handles_phase_tag_and_rejects_unknown_headers():
    item = parse_retrieval_block("[Jaen.pdf p.40 [Fase 2]]\nTexto")
    assert item.reference.source == "Jaen.pdf" and item.reference.page == "40"
    assert item.reference.reference_id == reference_id_for("Jaen.pdf", "40")
    assert parse_retrieval_block("[desconocido p.?]\nTexto") is None
    assert parse_retrieval_block("sin cabecera") is None
    assert parse_retrieval_block(None) is None


def test_packet_counts_distinct_source_pages_and_bounds_items():
    blocks = [
        "[A.pdf p.1]\nuno",
        "[A.pdf p.1]\nuno",  # exact duplicate dropped
        "[A.pdf p.1]\notro chunk misma página",
        "[B.pdf p.2]\n" + "x" * 50,
    ]
    packet = evidence_packet_from_blocks("q", blocks, max_item_chars=10)
    assert len(packet.items) == 3
    assert packet.distinct_reference_count == 2 and packet.is_sufficient
    assert all(len(item.content) <= 10 for item in packet.items)
    assert not evidence_packet_from_blocks("q", blocks[:3]).is_sufficient


def _call_response(signature="SIG"):
    return {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {"name": "generate_image", "args": {"prompt": "model"}},
                            "thoughtSignature": signature,
                        }
                    ],
                },
                "finishReason": "STOP",
            }
        ]
    }


def _request(messages):
    return ChatCompletionRequest.model_validate(
        {
            "model": "gemini-test",
            "messages": messages,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "generate_image",
                        "parameters": {"type": "object", "properties": {"prompt": {"type": "string"}}},
                    },
                }
            ],
        }
    )


def _continuation(call_id, arguments):
    return _request(
        [
            {"role": "user", "content": "draw"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": "generate_image", "arguments": json.dumps(arguments)},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": call_id, "content": "done"},
        ]
    )


@pytest.mark.asyncio
async def test_rebound_arguments_are_required_on_continuation_and_signature_is_kept():
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        if len(bodies) == 1:
            return httpx.Response(200, json=_call_response())
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"role": "model", "parts": [{"text": "ok"}]}, "finishReason": "STOP"}]},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = GeminiNativeAdapter(api_key="k", client=client, id_factory=lambda: "call_1")
        await adapter.generate(_request([{"role": "user", "content": "draw"}]))
        adapter.rebind_continuation_arguments("call_1", {"prompt": "canonical"})

        with pytest.raises(ContinuationMismatch):
            await adapter.generate(_continuation("call_1", {"prompt": "model"}))
        # A mismatch is detected before dispatch and does not consume state.
        assert adapter.pending_continuation_count == 1

        result = await adapter.generate(_continuation("call_1", {"prompt": "canonical"}))
        assert result.events[0].text == "ok"
        sent = bodies[-1]["contents"][1]["parts"][0]
        assert sent["thoughtSignature"] == "SIG"
        assert sent["functionCall"]["args"] == {"prompt": "model"}
        assert adapter.pending_continuation_count == 0


@pytest.mark.asyncio
async def test_pending_state_expires_and_can_be_discarded():
    now = [0.0]
    ids = iter(["call_a", "call_b"])
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=_call_response()))
    ) as client:
        adapter = GeminiNativeAdapter(
            api_key="k",
            client=client,
            id_factory=lambda: next(ids),
            pending_ttl_seconds=10,
            clock=lambda: now[0],
        )
        await adapter.generate(_request([{"role": "user", "content": "draw"}]))
        assert adapter.discard_continuation("call_a") is True
        assert adapter.discard_continuation("call_a") is False

        await adapter.generate(_request([{"role": "user", "content": "draw"}]))
        now[0] = 11.0
        assert adapter.pending_continuation_count == 0
        with pytest.raises(MissingContinuationState):
            adapter.rebind_continuation_arguments("call_b", {"prompt": "x"})
