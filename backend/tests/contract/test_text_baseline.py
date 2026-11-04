import json
from pathlib import Path

import httpx
import pytest


SNAPSHOT = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "openapi-text-baseline.json"
)


def _contract_projection(openapi):
    chat_operation = openapi["paths"]["/v1/chat/completions"]["post"]
    request_ref = chat_operation["requestBody"]["content"]["application/json"][
        "schema"
    ]["$ref"]
    schema_name = request_ref.rsplit("/", 1)[-1]
    request_schema = openapi["components"]["schemas"][schema_name]

    property_keys = {
        "model": ("type", "default"),
        "messages": ("type",),
        "temperature": ("type", "default"),
        "stream": ("type", "default"),
        "top_k": ("type", "default", "minimum", "maximum"),
    }
    properties = {
        name: {
            key: request_schema["properties"][name][key]
            for key in keys
        }
        for name, keys in property_keys.items()
    }

    return {
        "paths": {
            "/v1/models": {
                "method": "get",
                "responses": sorted(
                    openapi["paths"]["/v1/models"]["get"]["responses"]
                ),
            },
            "/v1/chat/completions": {
                "method": "post",
                "request_schema": schema_name,
                "responses": sorted(chat_operation["responses"]),
            },
        },
        "schemas": {
            schema_name: {
                "required": request_schema["required"],
                "properties": properties,
            }
        },
    }


@pytest.mark.asyncio
async def test_models_response_keeps_openai_list_shape(app_module, monkeypatch):
    async def fake_ollama_models():
        return [
            {
                "id": "local-test-model",
                "object": "model",
                "owned_by": "ollama",
                "created": 0,
            }
        ]

    monkeypatch.setattr(app_module, "_get_ollama_models", fake_ollama_models)
    transport = httpx.ASGITransport(app=app_module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/v1/models")

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "list"
    assert payload["data"][-1] == {
        "id": "local-test-model",
        "object": "model",
        "owned_by": "ollama",
        "created": 0,
    }
    assert all({"id", "object", "owned_by", "created"} <= model.keys() for model in payload["data"])


@pytest.mark.asyncio
async def test_text_chat_keeps_rag_sources_and_openai_response_shape(
    app_module, monkeypatch
):
    class EncodedBatch:
        def tolist(self):
            return [[0.1, 0.2]]

    class FakeEmbedder:
        def encode(self, values, normalize_embeddings=True):
            return EncodedBatch()

    class FakeCollection:
        def query(self, **kwargs):
            return {
                "documents": [["Contenido documental"]],
                "metadatas": [[{"source": "fuente.pdf", "page": 7}]],
                "distances": [[0.1]],
            }

    captured = {}

    async def fake_gemini(model, prompt, temperature):
        captured.update(model=model, prompt=prompt, temperature=temperature)
        return "Respuesta textual"

    monkeypatch.setattr(app_module, "get_embedder", lambda: FakeEmbedder())
    monkeypatch.setattr(app_module, "collection", FakeCollection())
    monkeypatch.setattr(
        app_module,
        "rerank_blocks",
        lambda query, blocks, keep_top_k=None, max_length=None: blocks,
    )
    monkeypatch.setattr(app_module, "call_gemini_chat_completions", fake_gemini)

    transport = httpx.ASGITransport(app=app_module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gemini-3.8-flash",
                "messages": [{"role": "user", "content": "Explicá el hallazgo"}],
                "temperature": 0.2,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "chat.completion"
    assert payload["model"] == "gemini-3.8-flash"
    assert payload["id"].startswith("chatcmpl-")
    assert isinstance(payload["created"], int)
    assert payload["choices"] == [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Respuesta textual\n\n---\n**Fuentes**\n- [fuente.pdf p.7]",
            },
            "finish_reason": "stop",
        }
    ]
    assert payload["usage"] == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    assert "Contenido documental" in captured["prompt"]
    assert "Explicá el hallazgo" in captured["prompt"]
    assert captured["temperature"] == 0.2


@pytest.mark.asyncio
async def test_text_chat_without_user_message_keeps_current_400(app_module):
    transport = httpx.ASGITransport(app=app_module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "system", "content": "Solo sistema"}]},
        )

    assert response.status_code == 400
    assert response.json() == {"detail": "No se encontró mensaje de usuario."}


def test_openapi_text_contract_matches_snapshot(app_module):
    expected = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    assert _contract_projection(app_module.app.openapi()) == expected
