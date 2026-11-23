import json

from backend.api.grounded_image import _RENDER_POLICY, _render_user_message


CANONICAL = json.dumps({"query": "genera una imagen del oppidum de Giribaile", "facts": []})


def test_caption_message_quotes_request_and_states_configured_language():
    message = _render_user_message(CANONICAL, "Spanish")
    assert 'Original user request (data): "genera una imagen del oppidum de Giribaile"' in message
    assert "Caption language: Spanish" in message
    assert message.endswith(CANONICAL)


def test_caption_language_defaults_to_the_request_language():
    message = _render_user_message(CANONICAL)
    assert "Caption language: the language of the original user request" in message


def test_caption_message_tolerates_missing_query():
    message = _render_user_message("{}", "Spanish")
    assert "Original user request" not in message
    assert "Caption language: Spanish" in message


def test_policy_points_to_the_stated_caption_language():
    assert "caption language stated in the user message" in _RENDER_POLICY
