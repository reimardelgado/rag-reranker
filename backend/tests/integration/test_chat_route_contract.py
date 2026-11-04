from copy import deepcopy
import json

import httpx
import pytest
from fastapi import FastAPI

from backend.api.chat import ChatDependencies, create_chat_router
from backend.config import GroundedImageAccessGate, GroundedImageConfig, ImageAccessDenied
from backend.providers.base import (
    FinishCategory,
    FinishEvent,
    FunctionCallEvent,
    ProviderMetadata,
    ProviderRateLimited,
    ProviderTimeout,
    ProviderUnavailable,
    TextEvent,
)


METADATA = ProviderMetadata(provider="offline-fixture", candidate_count=1)


class ScriptedAdapter:
    def __init__(self, factory):
        self.factory = factory
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        return self.factory()


def _finish(category=FinishCategory.STOP):
    return FinishEvent(category, "offline-finish", METADATA)


def _app(adapter=None, *, legacy_answer="text fallback"):
    queries = []

    async def no_models():
        return []

    async def legacy_chat(model, prompt, temperature):
        return legacy_answer

    def retrieve(query, top_k):
        queries.append((query, top_k))
        return []

    dependencies = ChatDependencies(
        get_gemini_models=lambda: [],
        get_ollama_models=no_models,
        retrieve_blocks=retrieve,
        is_ollama_model=lambda model: False,
        call_ollama_chat=lambda model, messages, temperature: legacy_chat(
            model, "", temperature
        ),
        call_gemini_chat=legacy_chat,
        logger=object(),
        provider_adapter=adapter,
    )
    app = FastAPI()
    routes = create_chat_router(
        dependencies, default_model="offline-model", default_top_k=6
    )
    app.include_router(routes.router)
    return app, queries


def _decode_sse(body):
    values = []
    for frame in body.split("\n\n"):
        if not frame:
            continue
        assert frame.startswith("data: ")
        value = frame[6:]
        values.append(value if value == "[DONE]" else json.loads(value))
    return values


def _stream_text(payloads):
    return "".join(
        payload["choices"][0]["delta"].get("content", "")
        for payload in payloads
        if payload != "[DONE]"
    )


@pytest.mark.asyncio
async def test_route_preserves_multimodal_request_and_response_mode_equivalence():
    async def response_events():
        yield TextEvent("deterministic ")
        yield TextEvent("answer")
        yield _finish()

    payload = {
        "model": "offline-model",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "first part"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,YQ=="},
                    },
                    {"type": "text", "text": "second part"},
                ],
                "client_extension": {"retained": True},
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup_fixture",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        "tool_choice": "auto",
        "provider_options": {"fixture": True},
    }
    original = deepcopy(payload)
    outputs = {}

    for stream in (False, True):
        adapter = ScriptedAdapter(response_events)
        app, queries = _app(adapter)
        request_payload = deepcopy(payload) | {"stream": stream}
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post("/v1/chat/completions", json=request_payload)

        assert response.status_code == 200
        assert queries == [("first part\nsecond part", 6)]
        assert adapter.requests[0].model_dump(exclude_unset=True) == request_payload
        if stream:
            decoded = _decode_sse(response.text)
            assert decoded.count("[DONE]") == 1
            assert decoded[-2]["choices"][0]["finish_reason"] == "stop"
            outputs[stream] = _stream_text(decoded)
        else:
            body = response.json()
            assert body["choices"][0]["finish_reason"] == "stop"
            outputs[stream] = body["choices"][0]["message"]["content"]

        assert "offline-fixture" not in response.text
        assert "offline-finish" not in response.text

    assert outputs == {False: "deterministic answer", True: "deterministic answer"}
    assert payload == original


@pytest.mark.asyncio
async def test_attachment_only_turn_is_rejected_before_retrieval_or_provider_execution():
    async def response_events():
        raise AssertionError("attachment-only input reached provider")
        yield  # pragma: no cover

    adapter = ScriptedAdapter(response_events)
    app, queries = _app(adapter)
    payload = {
        "model": "offline-model",
        "messages": [
            {"role": "user", "content": "stale text"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,YQ=="},
                    }
                ],
            },
        ],
    }

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 400
    assert queries == []
    assert adapter.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    ("failure", "status_code", "code"),
    [
        (ProviderTimeout, 504, "PROVIDER_TIMEOUT"),
        (ProviderRateLimited, 429, "RATE_LIMITED"),
        (ProviderUnavailable, 502, "PROVIDER_UNAVAILABLE"),
    ],
)
async def test_provider_failures_are_recoverable_before_output_in_both_modes(
    stream, failure, status_code, code
):
    async def failed_events():
        raise failure()
        yield  # pragma: no cover

    app, _ = _app(ScriptedAdapter(failed_events))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "offline-model",
                "messages": [{"role": "user", "content": "fixture request"}],
                "stream": stream,
            },
        )

    assert response.status_code == status_code
    assert response.json()["error"]["code"] == code
    assert "request_id" in response.json()


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_malformed_tool_arguments_fail_closed_without_tool_output(stream):
    async def malformed_events():
        yield FunctionCallEvent(
            0,
            "call_fixture",
            "generate_image",
            {"invalid": object()},
        )
        yield _finish(FinishCategory.TOOL_CALLS)

    app, _ = _app(ScriptedAdapter(malformed_events))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "offline-model",
                "messages": [{"role": "user", "content": "fixture request"}],
                "stream": stream,
            },
        )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_TOOL_CALL"
    assert "tool_calls" not in response.text


@pytest.mark.asyncio
async def test_disabled_image_rollback_denies_before_headers_and_keeps_text_available():
    class UnreadableHeaders(dict):
        def items(self):
            raise AssertionError("disabled rollback inspected trusted headers")

    gate = GroundedImageAccessGate(GroundedImageConfig.from_environ({}))
    with pytest.raises(ImageAccessDenied, match="IMAGE_DISABLED"):
        gate.authorize(
            UnreadableHeaders(),
            model_id="offline-model",
            permission_granted=True,
            capability_enabled=True,
            image_toggle_enabled=True,
        )

    app, queries = _app(legacy_answer="text fallback")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "offline-model",
                "messages": [{"role": "user", "content": "ordinary text"}],
            },
        )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "text fallback"
    assert queries == [("ordinary text", 6)]
