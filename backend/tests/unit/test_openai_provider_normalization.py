import json

import httpx
import pytest
from fastapi import FastAPI

from backend.api.chat import ChatDependencies, create_chat_router
from backend.chat_contracts import ChatCompletionRequest
from backend.providers.base import (
    FinishCategory,
    FinishEvent,
    FunctionCallEvent,
    OpenAINormalizationError,
    OpenAIResponseNormalizer,
    ProviderMetadata,
    ProviderRateLimited,
    ProviderTimeout,
    TextEvent,
)


METADATA = ProviderMetadata(
    provider="private-provider-marker",
    candidate_count=1,
    prompt_tokens=2,
    completion_tokens=3,
    total_tokens=5,
)


async def _events(*events):
    for event in events:
        yield event


def _finish(category=FinishCategory.STOP, native_reason="PRIVATE_NATIVE_REASON"):
    return FinishEvent(category, native_reason, METADATA)


def _decode_sse(chunks):
    decoded = []
    for chunk in chunks:
        line = chunk.decode().strip()
        assert line.startswith("data: ")
        value = line[6:]
        decoded.append(value if value == "[DONE]" else json.loads(value))
    return decoded


@pytest.mark.asyncio
async def test_text_stream_is_openai_compatible_and_has_one_done_without_metadata_leakage():
    normalizer = OpenAIResponseNormalizer()
    chunks = [
        chunk
        async for chunk in normalizer.stream(
            _events(TextEvent("hello "), TextEvent("world"), _finish()),
            model="test-model",
            response_id="chatcmpl-stable",
            created=123,
        )
    ]
    payloads = _decode_sse(chunks)

    assert payloads[-1] == "[DONE]"
    assert payloads.count("[DONE]") == 1
    assert {item["id"] for item in payloads[:-1]} == {"chatcmpl-stable"}
    assert payloads[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert payloads[-2]["choices"][0]["finish_reason"] == "stop"
    rendered = b"".join(chunks).decode()
    assert "private-provider-marker" not in rendered
    assert "PRIVATE_NATIVE_REASON" not in rendered


@pytest.mark.asyncio
async def test_tool_stream_normalizes_missing_index_and_fragments_valid_arguments():
    normalizer = OpenAIResponseNormalizer(argument_fragment_chars=9)
    call = FunctionCallEvent(
        None,  # type: ignore[arg-type] -- malformed provider index is normalized.
        "call_public_stable",
        "capture_probe",
        {"prompt": "bounded", "size": "small"},
    )
    chunks = [
        chunk
        async for chunk in normalizer.stream(
            _events(call, _finish(FinishCategory.TOOL_CALLS)),
            model="test-model",
            response_id="chatcmpl-tool",
            created=123,
        )
    ]
    payloads = _decode_sse(chunks)
    deltas = [item["choices"][0]["delta"] for item in payloads[:-1]]
    tool_deltas = [delta["tool_calls"][0] for delta in deltas if "tool_calls" in delta]

    assert {delta["index"] for delta in tool_deltas} == {0}
    assert tool_deltas[0]["id"] == "call_public_stable"
    assert tool_deltas[0]["function"]["name"] == "capture_probe"
    assert sum("name" in delta.get("function", {}) for delta in tool_deltas) == 1
    assembled = "".join(delta["function"].get("arguments", "") for delta in tool_deltas)
    assert json.loads(assembled) == {"prompt": "bounded", "size": "small"}
    assert payloads[-2]["choices"][0]["finish_reason"] == "tool_calls"
    assert payloads.count("[DONE]") == 1


@pytest.mark.asyncio
async def test_non_stream_tool_response_is_equivalent_and_text_remains_unchanged():
    normalizer = OpenAIResponseNormalizer()
    call = FunctionCallEvent(0, "call_public", "capture_probe", {"value": "safe"})
    tool = await normalizer.non_stream(
        _events(call, _finish(FinishCategory.TOOL_CALLS)),
        model="test-model",
        response_id="chatcmpl-tool",
        created=123,
    )
    text = await normalizer.non_stream(
        _events(TextEvent("ordinary text"), _finish()),
        model="test-model",
        response_id="chatcmpl-text",
        created=123,
    )

    choice = tool["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] is None
    assert choice["message"]["tool_calls"] == [
        {
            "id": "call_public",
            "type": "function",
            "function": {
                "name": "capture_probe",
                "arguments": '{"value":"safe"}',
            },
        }
    ]
    assert text["choices"][0]["message"]["content"] == "ordinary text"
    assert text["choices"][0]["finish_reason"] == "stop"


@pytest.mark.asyncio
async def test_malformed_arguments_fail_instead_of_releasing_a_tool_call():
    normalizer = OpenAIResponseNormalizer()
    malformed = FunctionCallEvent(0, "call_public", "capture_probe", {"bad": object()})
    with pytest.raises(OpenAINormalizationError):
        await normalizer.non_stream(
            _events(malformed, _finish(FinishCategory.TOOL_CALLS)), model="test-model"
        )
    with pytest.raises(OpenAINormalizationError):
        normalizer.validate_first_event(malformed)


class FakeAdapter:
    def __init__(self, factory):
        self.factory = factory
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        return self.factory()


def _app(adapter=None, normalizer=None, *, legacy_answer="legacy answer"):
    async def ollama_models():
        return []

    async def legacy_gemini(model, prompt, temperature):
        return legacy_answer

    dependencies = ChatDependencies(
        get_gemini_models=lambda: [],
        get_ollama_models=ollama_models,
        retrieve_blocks=lambda query, top_k: [],
        is_ollama_model=lambda model: False,
        call_ollama_chat=lambda model, messages, temperature: legacy_gemini(
            model, "", temperature
        ),
        call_gemini_chat=legacy_gemini,
        logger=object(),
        provider_adapter=adapter,
        response_normalizer=normalizer,
    )
    app = FastAPI()
    routes = create_chat_router(
        dependencies, default_model="test-model", default_top_k=6
    )
    app.state.chat_routes = routes
    app.include_router(routes.router)
    return app


@pytest.mark.asyncio
async def test_route_maps_error_before_stream_and_emits_typed_error_during_stream():
    async def before():
        raise ProviderTimeout()
        yield  # pragma: no cover

    transport = httpx.ASGITransport(app=_app(FakeAdapter(before)))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        )
    assert response.status_code == 504
    assert response.json()["error"]["code"] == "PROVIDER_TIMEOUT"

    async def during():
        yield TextEvent("partial")
        raise ProviderRateLimited()

    transport = httpx.ASGITransport(app=_app(FakeAdapter(during)))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        )
    assert response.status_code == 200
    assert response.text.count("data: [DONE]") == 1
    assert '"code":"RATE_LIMITED"' in response.text
    assert "partial" in response.text


@pytest.mark.asyncio
async def test_adapter_route_forwards_rich_tool_continuation_without_private_fields():
    async def final_text():
        yield TextEvent("continued")
        yield _finish()

    adapter = FakeAdapter(final_text)
    transport = httpx.ASGITransport(app=_app(adapter))
    payload = {
        "model": "test-model",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "prior"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,aA=="}}]},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "call_public", "type": "function", "function": {"name": "capture_probe", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call_public", "content": "{\"ok\":true}"},
        ],
        "tools": [{"type": "function", "function": {"name": "capture_probe", "parameters": {"type": "object"}}}],
        "tool_choice": "auto",
        "client_extension": {"preserved": True},
    }
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    forwarded = adapter.requests[0].model_dump(exclude_unset=True)
    assert forwarded == payload
    rendered = response.text
    assert "PRIVATE_NATIVE_REASON" not in rendered
    assert "private-provider-marker" not in rendered


@pytest.mark.asyncio
async def test_cancellation_closes_provider_iterator_and_legacy_route_stays_inactive():
    closed = False

    async def cancellable():
        nonlocal closed
        try:
            yield TextEvent("first")
            yield _finish()
        finally:
            closed = True

    adapter = FakeAdapter(cancellable)
    cancellation_app = _app(adapter)
    endpoint = cancellation_app.state.chat_routes.openai_chat_completions
    response = await endpoint(
        ChatCompletionRequest.model_validate(
            {
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            }
        )
    )
    await anext(response.body_iterator)
    await response.body_iterator.aclose()
    assert closed

    transport = httpx.ASGITransport(app=_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "legacy answer"
