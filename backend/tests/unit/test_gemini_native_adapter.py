import json
from copy import deepcopy

import httpx
import pytest

from backend.chat_contracts import ChatCompletionRequest
from backend.providers.base import (
    ContinuationMismatch,
    FinishCategory,
    FinishEvent,
    FunctionCallEvent,
    InvalidProviderRequest,
    MalformedProviderResponse,
    MissingContinuationState,
    ProviderRateLimited,
    ProviderTimeout,
    ProviderUnavailable,
    TextEvent,
)
from backend.providers.gemini_native import GeminiNativeAdapter


MODEL_RESPONSE = {
    "candidates": [
        {
            "content": {"role": "model", "parts": [{"text": "safe answer"}]},
            "finishReason": "STOP",
        }
    ],
    "usageMetadata": {
        "promptTokenCount": 4,
        "candidatesTokenCount": 2,
        "totalTokenCount": 6,
    },
}


def _request(**overrides):
    payload = {
        "model": "gemini-test",
        "messages": [{"role": "user", "content": "hello"}],
    }
    payload.update(overrides)
    return ChatCompletionRequest.model_validate(payload)


def _client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_provider_interface_yields_typed_events_and_terminal_finish():
    async with _client(lambda request: httpx.Response(200, json=MODEL_RESPONSE)) as client:
        adapter = GeminiNativeAdapter(api_key="test-key", client=client)
        events = [event async for event in adapter.complete(_request())]

    assert events[0] == TextEvent("safe answer")
    assert isinstance(events[-1], FinishEvent)
    assert events[-1].category is FinishCategory.STOP
    assert events[-1].metadata.total_tokens == 6


@pytest.mark.asyncio
async def test_converts_tools_choice_roles_and_multimodal_without_mutation():
    captured = []

    def handler(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json=MODEL_RESPONSE)

    payload = {
        "model": "gemini-test",
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "policy"}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "inspect"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,aGVsbG8="},
                    },
                    {
                        "type": "input_audio",
                        "input_audio": {"data": "aGVsbG8=", "format": "wav"},
                    },
                ],
            },
            {"role": "assistant", "content": "prior answer"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "capture_probe",
                    "description": "Capture a bounded value",
                    "parameters": {
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                        "required": ["value"],
                    },
                },
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": "capture_probe"}},
        "max_tokens": 20,
        "top_p": 0.8,
        "stop": ["END"],
    }
    original = deepcopy(payload)
    request = ChatCompletionRequest.model_validate(payload)
    before = request.model_dump(exclude_unset=True)

    async with _client(handler) as client:
        result = await GeminiNativeAdapter(api_key="test-key", client=client).generate(request)

    assert result.events == (TextEvent("safe answer"),)
    body = captured[0]
    assert body["systemInstruction"] == {"parts": [{"text": "policy"}]}
    assert body["contents"] == [
        {
            "role": "user",
            "parts": [
                {"text": "inspect"},
                {"inlineData": {"mimeType": "image/png", "data": "aGVsbG8="}},
                {"inlineData": {"mimeType": "audio/wav", "data": "aGVsbG8="}},
            ],
        },
        {"role": "model", "parts": [{"text": "prior answer"}]},
    ]
    assert body["tools"][0]["functionDeclarations"][0] == payload["tools"][0]["function"]
    assert body["toolConfig"] == {
        "functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": ["capture_probe"]}
    }
    assert body["generationConfig"] == {
        "temperature": 0.6,
        "maxOutputTokens": 20,
        "topP": 0.8,
        "stopSequences": ["END"],
    }
    assert request.model_dump(exclude_unset=True) == before
    assert payload == original


@pytest.mark.asyncio
async def test_parses_native_call_without_disclosing_signature_or_provider_id():
    native = {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "id": "provider-private-id",
                                "name": "capture_probe",
                                "args": {"value": "bounded"},
                            },
                            "thoughtSignature": "provider-private-signature",
                        }
                    ],
                },
                "finishReason": "STOP",
            }
        ]
    }

    async with _client(lambda request: httpx.Response(200, json=native)) as client:
        adapter = GeminiNativeAdapter(
            api_key="test-key", client=client, id_factory=lambda: "call_safe_1"
        )
        result = await adapter.generate(_request())

    assert result.events == (
        FunctionCallEvent(0, "call_safe_1", "capture_probe", {"value": "bounded"}),
    )
    assert result.finish.category is FinishCategory.TOOL_CALLS
    rendered = repr((adapter, result))
    assert "test-key" not in rendered
    assert "provider-private-id" not in rendered
    assert "provider-private-signature" not in rendered


@pytest.mark.asyncio
async def test_returns_exact_native_model_content_on_matching_continuation_once():
    requests = []
    first_content = {
        "role": "model",
        "parts": [
            {
                "functionCall": {
                    "id": "provider-private-id",
                    "name": "capture_probe",
                    "args": {"value": "bounded"},
                },
                "thoughtSignature": "provider-private-signature",
            }
        ],
    }

    def handler(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return httpx.Response(
                200,
                json={
                    "candidates": [{"content": first_content, "finishReason": "STOP"}]
                },
            )
        return httpx.Response(200, json=MODEL_RESPONSE)

    async with _client(handler) as client:
        adapter = GeminiNativeAdapter(
            api_key="test-key", client=client, id_factory=lambda: "call_safe_1"
        )
        first = await adapter.generate(_request())
        call = first.events[0]
        continuation = _request(
            messages=[
                {"role": "user", "content": "hello"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call.tool_call_id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(call.arguments),
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": call.tool_call_id,
                    "content": json.dumps({"ok": True}),
                },
            ]
        )
        final = await adapter.generate(continuation)
        with pytest.raises(MissingContinuationState):
            await adapter.generate(continuation)

    assert final.events == (TextEvent("safe answer"),)
    assert requests[1]["contents"][1] == first_content
    assert requests[1]["contents"][2] == {
        "role": "user",
        "parts": [
            {
                "functionResponse": {
                    "name": "capture_probe",
                    "response": {"ok": True},
                    "id": "provider-private-id",
                }
            }
        ],
    }
    assert len(requests) == 2


@pytest.mark.asyncio
async def test_rejects_missing_or_mismatched_continuation_before_dispatch():
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=MODEL_RESPONSE)

    missing = _request(
        messages=[
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_unknown",
                        "type": "function",
                        "function": {"name": "capture_probe", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_unknown", "content": "{}"},
        ]
    )
    async with _client(handler) as client:
        adapter = GeminiNativeAdapter(api_key="test-key", client=client)
        with pytest.raises(MissingContinuationState):
            await adapter.generate(missing)
        with pytest.raises(ContinuationMismatch):
            await adapter.generate(
                _request(messages=[{"role": "tool", "tool_call_id": "wrong", "content": "{}"}])
            )
    assert calls == 0


@pytest.mark.asyncio
async def test_rejects_missing_native_signature_as_malformed():
    unsigned = {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "name": "capture_probe",
                                "args": {"value": "bounded"},
                            }
                        }
                    ],
                },
                "finishReason": "STOP",
            }
        ]
    }
    async with _client(lambda request: httpx.Response(200, json=unsigned)) as client:
        with pytest.raises(MalformedProviderResponse) as raised:
            await GeminiNativeAdapter(api_key="test-key", client=client).generate(_request())
    assert raised.value.code == "PROVIDER_INCOMPATIBLE"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "error"),
    [(429, ProviderRateLimited), (500, ProviderUnavailable), (503, ProviderUnavailable)],
)
async def test_maps_rate_limit_and_server_failures_without_provider_body(status, error):
    async with _client(
        lambda request: httpx.Response(status, content=b'private provider body')
    ) as client:
        adapter = GeminiNativeAdapter(api_key="test-key", client=client)
        with pytest.raises(error) as raised:
            await adapter.generate(_request())
    assert "private provider body" not in repr(raised.value)


@pytest.mark.asyncio
async def test_maps_timeout_and_rejects_malformed_or_oversized_responses():
    def timeout(request):
        raise httpx.ReadTimeout("private timeout detail", request=request)

    async with _client(timeout) as client:
        with pytest.raises(ProviderTimeout) as raised:
            await GeminiNativeAdapter(api_key="test-key", client=client).generate(_request())
    assert "private timeout detail" not in repr(raised.value)

    for response, limit in (
        (httpx.Response(200, content=b"not-json"), 1024),
        (httpx.Response(200, content=b"{}" * 20), 8),
    ):
        async with _client(lambda request, response=response: response) as client:
            with pytest.raises(MalformedProviderResponse):
                await GeminiNativeAdapter(
                    api_key="test-key", client=client, max_response_bytes=limit
                ).generate(_request())


@pytest.mark.asyncio
async def test_rejects_ambiguous_or_unsupported_content_before_dispatch():
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=MODEL_RESPONSE)

    requests = [
        _request(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.test/private.png"},
                        }
                    ],
                }
            ]
        ),
        _request(
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "file", "file": {"file_id": "opaque"}}],
                }
            ]
        ),
        _request(provider_options={"thinking": {"budget": 10}}),
    ]
    async with _client(handler) as client:
        adapter = GeminiNativeAdapter(api_key="test-key", client=client)
        for request in requests:
            with pytest.raises(InvalidProviderRequest):
                await adapter.generate(request)
    assert calls == 0
