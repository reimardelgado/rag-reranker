from copy import deepcopy

from backend.chat_contracts import (
    ChatCompletionRequest,
    OpenAIChatRequest,
    extract_user_query,
)


def test_chat_contract_round_trips_roles_tools_and_unknown_fields_losslessly():
    payload = {
        "model": "adapter-model",
        "messages": [
            {"role": "system", "content": "policy", "provider_tag": "s1"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe ", "cache_control": {"type": "ephemeral"}},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc", "detail": "low"}},
                    {"type": "text", "text": "this site"},
                ],
                "client_message_id": "u1",
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "generate_image", "arguments": "{\"prompt\":\"site\"}"},
                        "thought_signature": "opaque-signature",
                    }
                ],
            },
            {
                "role": "tool",
                "content": "image-id-1",
                "tool_call_id": "call_1",
                "provider_status": "stored",
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "generate_image",
                    "description": "Generate a grounded image",
                    "parameters": {"type": "object", "properties": {"prompt": {"type": "string"}}},
                    "x-provider-schema": "v2",
                },
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": "generate_image"}},
        "stream": True,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
        "parallel_tool_calls": False,
        "provider_options": {"thinking": {"budget": 128}},
    }

    request = ChatCompletionRequest.model_validate(payload)

    assert OpenAIChatRequest is ChatCompletionRequest
    assert request.model_dump(exclude_unset=True) == payload


def test_extract_user_query_collects_all_text_parts_without_mutation():
    payload = {
        "model": "adapter-model",
        "messages": [
            {"role": "user", "content": "earlier turn"},
            {"role": "assistant", "content": "answer"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "first textual part"},
                    {"type": "image_url", "image_url": {"url": "https://example.test/site.png"}},
                    {"type": "text", "text": "second textual part", "future": "kept"},
                ],
            },
        ],
    }
    original = deepcopy(payload)
    request = ChatCompletionRequest.model_validate(payload)
    model_before = request.model_dump(exclude_unset=True)

    assert extract_user_query(request.messages) == "first textual part\nsecond textual part"
    assert request.model_dump(exclude_unset=True) == model_before
    assert payload == original


def test_attachment_only_latest_turn_has_no_retrieval_query():
    messages = [
        {"role": "user", "content": "do not reuse this stale query"},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "https://example.test/site.png"}},
                {"type": "file", "file": {"file_id": "file-1"}},
            ],
        },
    ]

    request = ChatCompletionRequest.model_validate(
        {"model": "adapter-model", "messages": messages}
    )

    assert extract_user_query(request.messages) == ""
    assert extract_user_query(messages) == ""
