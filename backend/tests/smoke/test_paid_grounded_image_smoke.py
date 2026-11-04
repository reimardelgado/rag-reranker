"""Opt-in live smoke of the grounded image loop against a running backend.

Double-guarded: excluded by the default ``-m "not paid"`` selection AND
skipped unless ``RAG_PAID_SMOKE=1``.  It calls the live Gemini text API
(tokens cost money) through the deployed backend.  It does NOT invoke an image
engine: round two uses a synthetic tool result.  Real image generation is
verified manually through the Open WebUI canary described in
``docs/grounded-image-activation.md``.

Run only with explicit approval::

    RAG_PAID_SMOKE=1 RAG_SMOKE_BASE_URL=http://localhost:8088 \
    RAG_SMOKE_USER=<canary-user-id> RAG_TRUSTED_SERVICE_TOKEN=<token> \
    pytest -m paid backend/tests/smoke -q
"""

import json
import os
import uuid

import httpx
import pytest

pytestmark = [
    pytest.mark.paid,
    pytest.mark.skipif(
        os.environ.get("RAG_PAID_SMOKE") != "1",
        reason="Live paid smoke requires explicit RAG_PAID_SMOKE=1 approval.",
    ),
]

IMAGE_TOOL = {
    "type": "function",
    "function": {
        "name": "generate_image",
        "parameters": {
            "type": "object",
            "properties": {"prompt": {"type": "string"}},
            "required": ["prompt"],
        },
    },
}


def test_live_two_round_grounded_loop():
    base_url = os.environ["RAG_SMOKE_BASE_URL"]
    model = os.environ.get("RAG_SMOKE_MODEL", "gemini-3.8-flash")
    headers = {
        "Authorization": f"Bearer {os.environ['RAG_TRUSTED_SERVICE_TOKEN']}",
        "X-OpenWebUI-User-Id": os.environ["RAG_SMOKE_USER"],
        "X-OpenWebUI-Chat-Id": f"smoke-{uuid.uuid4().hex}",
        "X-OpenWebUI-Message-Id": f"smoke-{uuid.uuid4().hex}",
    }
    first = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": os.environ.get(
                    "RAG_SMOKE_QUERY", "Genera una imagen de un oppidum ibero amurallado"
                ),
            }
        ],
        "tools": [IMAGE_TOOL],
    }
    with httpx.Client(base_url=base_url, timeout=120) as client:
        response = client.post("/v1/chat/completions", json=first, headers=headers)
        assert response.status_code == 200, response.text
        [call] = response.json()["choices"][0]["message"]["tool_calls"]
        assert json.loads(json.loads(call["function"]["arguments"])["prompt"])["evidence_refs"]

        second = dict(
            first,
            messages=[
                *first["messages"],
                {"role": "assistant", "content": "", "tool_calls": [call]},
                {"role": "tool", "tool_call_id": call["id"], "content": "smoke: synthetic result"},
            ],
        )
        response = client.post("/v1/chat/completions", json=second, headers=headers)
        assert response.status_code == 200, response.text
        assert "Interpretive reconstruction" in response.json()["choices"][0]["message"]["content"]
