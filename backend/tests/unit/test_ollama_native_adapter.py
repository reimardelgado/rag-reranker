import json

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
    ProviderTimeout,
    ProviderUnavailable,
    TextEvent,
)
from backend.providers.ollama_native import OllamaNativeAdapter


TOOL = {
    "type": "function",
    "function": {
        "name": "generate_image",
        "description": "Generate an image.",
        "parameters": {
            "type": "object",
            "properties": {"prompt": {"type": "string"}},
            "required": ["prompt"],
        },
    },
}
FORCE = {"type": "function", "function": {"name": "generate_image"}}


def _reply(content="", tool_calls=None, done_reason="stop", **extra):
    message = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {
        "model": "qwen2.5:3b",
        "message": message,
        "done": True,
        "done_reason": done_reason,
        "prompt_eval_count": 7,
        "eval_count": 3,
        **extra,
    }


def _request(**overrides):
    payload = {"model": "qwen2.5:3b", "messages": [{"role": "user", "content": "hola"}]}
    payload.update(overrides)
    return ChatCompletionRequest.model_validate(payload)


class Recorder:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.bodies = []
        self.urls = []

    def __call__(self, request):
        self.bodies.append(json.loads(request.content))
        self.urls.append(str(request.url))
        response = self.responses.pop(0)
        return response if isinstance(response, httpx.Response) else httpx.Response(200, json=response)


def _adapter(recorder, **kwargs):
    client = httpx.AsyncClient(transport=httpx.MockTransport(recorder))
    ids = iter(f"call_{n}" for n in range(100))
    return OllamaNativeAdapter(
        client=client, base_url="http://ollama:11434/", id_factory=lambda: next(ids), **kwargs
    )


async def _events(adapter, request):
    return [event async for event in adapter.complete(request)]


@pytest.mark.asyncio
async def test_plain_text_turn_uses_native_chat_with_options_and_usage():
    recorder = Recorder(_reply("respuesta"))
    adapter = _adapter(recorder, keep_alive="2m")

    events = await _events(adapter, _request(temperature=0.3, max_tokens=50, stop="FIN"))

    assert recorder.urls == ["http://ollama:11434/api/chat"]
    body = recorder.bodies[0]
    assert body["stream"] is False and body["keep_alive"] == "2m"
    assert body["options"] == {"temperature": 0.3, "num_predict": 50, "stop": ["FIN"]}
    assert "tools" not in body and "format" not in body
    assert events[0] == TextEvent("respuesta")
    assert events[-1].category is FinishCategory.STOP
    assert events[-1].metadata.provider == "ollama"
    assert events[-1].metadata.total_tokens == 10


@pytest.mark.asyncio
async def test_forced_function_becomes_structured_output_without_tools():
    recorder = Recorder(_reply('{"prompt": "un oppidum"}'))
    adapter = _adapter(recorder)

    events = await _events(adapter, _request(tools=[TOOL], tool_choice=FORCE))

    body = recorder.bodies[0]
    assert "tools" not in body
    assert body["format"] == TOOL["function"]["parameters"]
    assert body["messages"][0]["role"] == "system"
    assert "generate_image" in body["messages"][0]["content"]
    [call] = [event for event in events if isinstance(event, FunctionCallEvent)]
    assert call.name == "generate_image" and call.arguments == {"prompt": "un oppidum"}
    assert call.tool_call_id == "call_0"
    assert events[-1].category is FinishCategory.TOOL_CALLS
    assert adapter.pending_continuation_count == 1


@pytest.mark.asyncio
async def test_forced_instruction_is_appended_to_existing_system_message():
    recorder = Recorder(_reply('```json\n{"prompt": "x"}\n```'))
    adapter = _adapter(recorder)
    request = _request(
        messages=[{"role": "system", "content": "policy"}, {"role": "user", "content": "go"}],
        tools=[TOOL],
        tool_choice="required",
    )

    events = await _events(adapter, request)

    system = recorder.bodies[0]["messages"][0]
    assert system["content"].startswith("policy\n\n")
    assert len(recorder.bodies[0]["messages"]) == 2
    assert isinstance(events[0], FunctionCallEvent) and events[0].arguments == {"prompt": "x"}


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["not json", "[1, 2]", ""])
async def test_forced_output_that_is_not_an_object_fails_closed(content):
    adapter = _adapter(Recorder(_reply(content)))
    with pytest.raises(MalformedProviderResponse):
        await _events(adapter, _request(tools=[TOOL], tool_choice=FORCE))
    assert adapter.pending_continuation_count == 0


@pytest.mark.asyncio
async def test_auto_tools_parse_native_tool_calls_with_object_or_string_arguments():
    recorder = Recorder(
        _reply(
            "",
            tool_calls=[
                {"function": {"name": "generate_image", "arguments": {"prompt": "a"}}},
                {"function": {"name": "generate_image", "arguments": '{"prompt": "b"}', "index": 1}},
            ],
        )
    )
    adapter = _adapter(recorder)

    events = await _events(adapter, _request(tools=[TOOL]))

    assert recorder.bodies[0]["tools"] == [TOOL]
    calls = [event for event in events if isinstance(event, FunctionCallEvent)]
    assert [call.arguments for call in calls] == [{"prompt": "a"}, {"prompt": "b"}]
    assert events[-1].category is FinishCategory.TOOL_CALLS


@pytest.mark.asyncio
async def test_think_block_is_stripped_from_text():
    adapter = _adapter(Recorder(_reply("<think>razonando</think>\nRespuesta final")))
    events = await _events(adapter, _request())
    assert events[0] == TextEvent("Respuesta final")


async def _first_round(adapter):
    events = await _events(adapter, _request(tools=[TOOL], tool_choice=FORCE))
    return next(event for event in events if isinstance(event, FunctionCallEvent))


def _continuation(call_id, arguments, *, tool_choice="none"):
    return _request(
        messages=[
            {"role": "user", "content": "go"},
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
            {"role": "tool", "tool_call_id": call_id, "content": '{"status": "ok"}'},
        ],
        tools=[TOOL],
        tool_choice=tool_choice,
    )


@pytest.mark.asyncio
async def test_rebound_continuation_is_rendered_as_text_when_tools_are_off():
    recorder = Recorder(_reply('{"prompt": "model"}'), _reply("Pie de imagen"))
    adapter = _adapter(recorder)
    call = await _first_round(adapter)
    adapter.rebind_continuation_arguments(call.tool_call_id, {"prompt": "canonical"})

    events = await _events(adapter, _continuation(call.tool_call_id, {"prompt": "canonical"}))

    body = recorder.bodies[1]
    assert "tools" not in body and "format" not in body
    roles = [message["role"] for message in body["messages"]]
    assert roles == ["user", "assistant", "user"]
    assert "canonical" in body["messages"][1]["content"]
    assert '{"status": "ok"}' in body["messages"][2]["content"]
    assert events[0] == TextEvent("Pie de imagen")
    assert adapter.pending_continuation_count == 0


@pytest.mark.asyncio
async def test_continuation_with_native_tools_uses_tool_role_messages():
    recorder = Recorder(_reply('{"prompt": "p"}'), _reply("ok"))
    adapter = _adapter(recorder)
    call = await _first_round(adapter)

    await _events(adapter, _continuation(call.tool_call_id, {"prompt": "p"}, tool_choice="auto"))

    messages = recorder.bodies[1]["messages"]
    assert messages[1]["tool_calls"] == [{"function": {"name": "generate_image", "arguments": {"prompt": "p"}}}]
    assert messages[2] == {"role": "tool", "content": '{"status": "ok"}', "tool_name": "generate_image"}


@pytest.mark.asyncio
async def test_continuation_rejects_tampered_arguments_and_unknown_ids():
    adapter = _adapter(Recorder(_reply('{"prompt": "p"}')))
    call = await _first_round(adapter)

    with pytest.raises(ContinuationMismatch):
        await _events(adapter, _continuation(call.tool_call_id, {"prompt": "other"}))
    with pytest.raises(MissingContinuationState):
        await _events(adapter, _continuation("call_unknown", {"prompt": "p"}))


@pytest.mark.asyncio
async def test_continuation_is_single_use():
    recorder = Recorder(_reply('{"prompt": "p"}'), _reply("ok"))
    adapter = _adapter(recorder)
    call = await _first_round(adapter)
    await _events(adapter, _continuation(call.tool_call_id, {"prompt": "p"}))

    with pytest.raises(MissingContinuationState):
        await _events(adapter, _continuation(call.tool_call_id, {"prompt": "p"}))


@pytest.mark.asyncio
async def test_pending_state_expires():
    now = [0.0]
    recorder = Recorder(_reply('{"prompt": "p"}'))
    client = httpx.AsyncClient(transport=httpx.MockTransport(recorder))
    adapter = OllamaNativeAdapter(client=client, pending_ttl_seconds=10, clock=lambda: now[0])
    await _first_round(adapter)
    assert adapter.pending_continuation_count == 1
    now[0] = 11.0
    assert adapter.pending_continuation_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response, error",
    [
        (httpx.Response(404, json={"error": "model not found"}), ProviderUnavailable),
        (httpx.Response(500, json={}), ProviderUnavailable),
        (httpx.Response(400, json={}), InvalidProviderRequest),
        (httpx.Response(200, content=b"not json"), MalformedProviderResponse),
        (httpx.Response(200, json={"done": False}), MalformedProviderResponse),
    ],
)
async def test_http_failures_map_to_safe_provider_errors(response, error):
    adapter = _adapter(Recorder(response))
    with pytest.raises(error):
        await _events(adapter, _request())


@pytest.mark.asyncio
async def test_timeout_and_transport_errors():
    def timeout(request):
        raise httpx.ReadTimeout("slow", request=request)

    def down(request):
        raise httpx.ConnectError("down", request=request)

    for handler, error in ((timeout, ProviderTimeout), (down, ProviderUnavailable)):
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        with pytest.raises(error):
            await _events(OllamaNativeAdapter(client=client), _request())


@pytest.mark.asyncio
async def test_rejects_unsupported_requests_before_dispatch():
    recorder = Recorder()
    adapter = _adapter(recorder)
    other = {"type": "function", "function": {"name": "other"}}
    with pytest.raises(InvalidProviderRequest):
        await _events(adapter, _request(tools=[TOOL, other], tool_choice="required"))
    with pytest.raises(InvalidProviderRequest):
        await _events(
            adapter, _request(tools=[TOOL], tool_choice={"type": "function", "function": {"name": "nope"}})
        )
    with pytest.raises(InvalidProviderRequest):
        await _events(
            adapter,
            _request(
                messages=[
                    {"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": "", "format": "wav"}}]}
                ]
            ),
        )
    with pytest.raises(ContinuationMismatch):
        await _events(
            adapter, _request(messages=[{"role": "tool", "tool_call_id": "x", "content": "y"}])
        )
    assert recorder.bodies == []


@pytest.mark.asyncio
async def test_image_data_urls_become_native_images():
    recorder = Recorder(_reply("veo una muralla"))
    adapter = _adapter(recorder)
    request = _request(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}},
                ],
            }
        ]
    )

    await _events(adapter, request)

    assert recorder.bodies[0]["messages"] == [
        {"role": "user", "content": "describe", "images": ["aGVsbG8="]}
    ]


@pytest.mark.asyncio
async def test_length_finish_and_finish_event_is_last():
    adapter = _adapter(Recorder(_reply("corto", done_reason="length")))
    events = await _events(adapter, _request())
    assert isinstance(events[-1], FinishEvent) and events[-1].category is FinishCategory.LENGTH
