"""Offline tests of the fully local mode (DEFAULT_PROVIDER=ollama).

The grounded image loop runs end to end through the real ``OllamaNativeAdapter``
against a scripted ``/api/chat`` endpoint: builder and render rounds arrive as
structured-output requests (``format``), the caption round without tools.
"""

from __future__ import annotations

import base64
import json

import httpx
import pytest
from fastapi import FastAPI

from backend.api.chat import ChatDependencies, create_chat_router
from backend.api.grounded_image import GroundedImageOrchestrator, GroundedImageRuntime
from backend.config import GroundedImageConfig
from backend.grounding.provenance import ADJACENT_LABEL, DOCUMENTARY_DISCLAIMER
from backend.grounding.retrieval_evidence import reference_id_for
from backend.metrics import GroundedImageMetrics, RolloutStage
from backend.providers.ollama_native import OllamaNativeAdapter


SERVICE_TOKEN = "svc-" + "t" * 40
MODEL = "qwen2.5:3b"
USER = "canary-user"
BLOCKS = [
    "[Jaen.pdf p.12]\nEl oppidum ibero estaba rodeado por una muralla de mampostería.",
    "[Jaen.pdf p.40]\nLas viviendas del oppidum ibero eran rectangulares con zócalo de piedra.",
]
REF_12 = reference_id_for("Jaen.pdf", "12")
REF_40 = reference_id_for("Jaen.pdf", "40")
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


class OllamaScript:
    def __init__(self):
        self.bodies: list[dict] = []

    def kinds(self):
        return [self._kind(body) for body in self.bodies]

    @staticmethod
    def _kind(body):
        properties = (body.get("format") or {}).get("properties", {})
        if "documented_facts" in properties:
            return "builder"
        if "prompt" in properties:
            return "render"
        if "tools" not in body and "format" not in body:
            return "continuation"
        return "other"

    def __call__(self, request):
        body = json.loads(request.content)
        self.bodies.append(body)
        kind = self._kind(body)
        if kind == "builder":
            content = json.dumps(
                {
                    "documented_facts": [
                        {"text": "Muralla de mampostería alrededor del oppidum", "reference_ids": [REF_12]},
                        {"text": "Viviendas rectangulares con zócalo de piedra", "reference_ids": [REF_40]},
                    ],
                    "inferred_elements": ["Cielo despejado"],
                    "uncertainties": ["Altura exacta de la muralla"],
                    "forbidden_additions": ["Castillo medieval"],
                }
            )
        elif kind == "render":
            content = json.dumps({"prompt": "model prompt that must be replaced"})
        elif kind == "continuation":
            content = "Reconstrucción interpretativa del oppidum amurallado."
        else:
            return httpx.Response(400, json={"error": "unexpected"})
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": content}, "done": True, "done_reason": "stop"},
        )


def _build():
    script = OllamaScript()
    config = GroundedImageConfig(
        enabled=True,
        trusted_boundary=True,
        provenance_enabled=True,
        adapter="ollama",
        authorized_users=frozenset({USER}),
        authorized_models=frozenset({MODEL}),
        service_token=SERVICE_TOKEN,
        correlation_hmac_key=b"k" * 32,
        chat_model=MODEL,
        image_model_id="comfyui-sd15",
    )
    adapter = OllamaNativeAdapter(client=httpx.AsyncClient(transport=httpx.MockTransport(script)))
    orchestrator = GroundedImageOrchestrator(
        GroundedImageRuntime(
            config=config,
            adapter=adapter,
            retrieve_blocks=lambda query, top_k: list(BLOCKS),
            retrieval_model_id="test-embedder",
            metrics=GroundedImageMetrics(RolloutStage.CANARY),
        )
    )

    async def no_models():
        return []

    async def no_text(*args):
        raise AssertionError("text path must not run")

    dependencies = ChatDependencies(
        get_gemini_models=lambda: [],
        get_ollama_models=no_models,
        retrieve_blocks=lambda query, top_k: [],
        is_ollama_model=lambda model: True,
        call_ollama_chat=no_text,
        call_gemini_chat=no_text,
        logger=object(),
        image_orchestrator=orchestrator,
    )
    app = FastAPI()
    app.include_router(create_chat_router(dependencies, default_model=MODEL, default_top_k=6).router)
    return app, script, adapter


def _headers():
    return {
        "Authorization": f"Bearer {SERVICE_TOKEN}",
        "X-OpenWebUI-User-Id": USER,
        "X-OpenWebUI-Chat-Id": "chat-1",
        "X-OpenWebUI-Message-Id": "msg-1",
    }


async def _post(app, payload):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post("/v1/chat/completions", json=payload, headers=_headers())


@pytest.mark.asyncio
async def test_local_two_round_loop_is_grounded_and_labelled():
    app, script, adapter = _build()
    first = {
        "model": MODEL,
        "stream": False,
        "messages": [{"role": "user", "content": "Genera una imagen del oppidum ibero"}],
        "tools": [IMAGE_TOOL],
    }

    response = await _post(app, first)
    assert response.status_code == 200, response.text
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    [tool_call] = choice["message"]["tool_calls"]
    arguments = json.loads(tool_call["function"]["arguments"])
    canonical = json.loads(arguments["prompt"])
    assert {ref["reference_id"] for ref in canonical["evidence_refs"]} == {REF_12, REF_40}
    assert "model prompt" not in arguments["prompt"]
    assert script.kinds() == ["builder", "render"]
    assert all("tools" not in body for body in script.bodies)

    second = {
        "model": MODEL,
        "stream": False,
        "messages": [
            *first["messages"],
            {"role": "assistant", "content": "", "tool_calls": [tool_call]},
            {
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": "![img](data:image/png;base64," + base64.b64encode(b"x" * 32).decode() + ")",
            },
        ],
        "tools": [IMAGE_TOOL],
    }
    response = await _post(app, second)
    assert response.status_code == 200, response.text
    content = response.json()["choices"][0]["message"]["content"]
    assert content.startswith("Reconstrucción interpretativa")
    assert ADJACENT_LABEL in content and DOCUMENTARY_DISCLAIMER in content
    assert script.kinds() == ["builder", "render", "continuation"]
    assert "base64" not in json.dumps(script.bodies[-1])
    assert adapter.pending_continuation_count == 0


def test_local_app_starts_without_gemini_key_and_lists_only_local_models(local_app_module):
    assert local_app_module.LLM_MODE.is_local
    assert local_app_module.DEFAULT_CHAT_MODEL == "qwen2.5:3b"
    assert local_app_module._get_gemini_models() == []
    assert local_app_module._is_ollama_model("gemini-3.8-flash")


@pytest.mark.asyncio
async def test_local_text_chat_maps_gemini_model_names_to_local_default(local_app_module, monkeypatch):
    seen = []

    async def fake_ollama(model, messages, temperature, stream=False):
        seen.append(model)
        return "respuesta local"

    async def fail_gemini(*args, **kwargs):
        raise AssertionError("Gemini must never be called in local mode")

    monkeypatch.setattr(local_app_module, "_call_ollama_chat", fake_ollama)
    monkeypatch.setattr(local_app_module, "call_gemini_chat_completions", fail_gemini)

    transport = httpx.ASGITransport(app=local_app_module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "gemini-3.8-flash", "messages": [{"role": "user", "content": "hola"}]},
        )

    assert response.status_code == 200, response.text
    assert seen == ["qwen2.5:3b"]
    assert "respuesta local" in response.json()["choices"][0]["message"]["content"]
