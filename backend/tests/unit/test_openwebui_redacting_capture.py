import http.client
import json
import sqlite3
import threading
from email.message import Message
from pathlib import Path

import pytest

from backend.tests.fixtures.openwebui.redacting_capture import (
    CaptureState,
    ThreadingHTTPServer,
    header_presence,
    make_handler,
    structural_shape,
)
from backend.tests.fixtures.openwebui.runtime_connection import (
    CONFIG_KEY,
    backup_and_redirect,
    rollback,
)


FORBIDDEN_MARKERS = (
    "Bearer ",
    "secret-api-key",
    "private prompt",
    "person@example.com",
    "chat-private-id",
    "tool-call-private-id",
)


def test_structural_shape_keeps_only_types_lengths_booleans_and_allowlisted_enums():
    source = {
        "model": "private-model-id",
        "stream": True,
        "messages": [
            {
                "role": "user",
                "content": "private prompt",
                "email": "person@example.com",
                "chat_id": "chat-private-id",
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "tool-call-private-id",
                        "function": {
                            "name": "generate_image",
                            "arguments": "secret image prompt",
                        },
                    }
                ],
            },
        ],
        "features": {"image_generation": True},
        "unsafe key person@example.com": "secret-api-key",
    }

    shape = structural_shape(source)
    serialized = json.dumps(shape)

    assert shape["keys"]["stream"] == {"type": "boolean", "safe_value": True}
    assert shape["keys"]["messages"]["length"] == 2
    assert shape["keys"]["messages"]["items"][0]["keys"]["role"]["safe_value"] == "user"
    function = shape["keys"]["messages"]["items"][1]["keys"]["tool_calls"]["items"][0]["keys"]["function"]
    assert function["keys"]["name"]["safe_value"] == "generate_image"
    assert "<redacted-key>" in shape["keys"]
    assert not any(marker in serialized for marker in FORBIDDEN_MARKERS)


def test_header_capture_records_names_and_never_values():
    headers = Message()
    headers["Authorization"] = "Bearer secret-api-key"
    headers["Cookie"] = "session=private"
    headers["Content-Type"] = "application/json"

    captured = header_presence(headers)

    assert captured == {
        "present_names": ["Authorization", "Content-Type", "Cookie"],
        "values_captured": False,
    }
    assert "secret" not in json.dumps(captured).lower()


@pytest.fixture
def capture_server(tmp_path):
    state = CaptureState(tmp_path / "capture.json")
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, state.output_path
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _request(server, method, path, body=None, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=2)
    payload = None if body is None else json.dumps(body)
    connection.request(method, path, body=payload, headers=headers or {})
    response = connection.getresponse()
    content = response.read()
    connection.close()
    return response.status, response.getheader("Content-Type"), content


def test_capture_server_supports_models_one_streaming_chat_and_rejects_extras(capture_server):
    server, output_path = capture_server
    status, _, models = _request(server, "GET", "/v1/models")
    assert status == 200
    assert json.loads(models)["data"][0]["id"] == "capture-safe-model"

    request = {
        "model": "private-model-id",
        "messages": [{"role": "user", "content": "private prompt"}],
        "tools": [{"type": "function", "function": {"name": "generate_image"}}],
        "stream": True,
    }
    status, content_type, response = _request(
        server,
        "POST",
        "/v1/chat/completions",
        request,
        {"Content-Type": "application/json", "Authorization": "Bearer secret-api-key"},
    )
    assert status == 200
    assert content_type == "text/event-stream"
    assert response.endswith(b"data: [DONE]\n\n")

    capture_text = output_path.read_text(encoding="utf-8")
    capture = json.loads(capture_text)
    assert capture["forwarded"] is False
    assert capture["accepted_chat_requests"] == 1
    assert capture["request"]["body"]["keys"]["stream"]["safe_value"] is True
    assert capture["request"]["headers"]["present_names"] == [
        "Accept-Encoding",
        "Authorization",
        "Content-Length",
        "Content-Type",
        "Host",
    ]
    assert not any(marker in capture_text for marker in FORBIDDEN_MARKERS)

    status, _, response = _request(server, "POST", "/v1/chat/completions", request)
    assert status == 409
    assert json.loads(response)["error"]["type"] == "capture_already_complete"


def _create_config_db(path: Path, urls: list[str]) -> None:
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE config (key TEXT PRIMARY KEY, value JSON NOT NULL)")
    connection.execute(
        "INSERT INTO config(key, value) VALUES (?, ?)",
        (CONFIG_KEY, json.dumps(urls)),
    )
    connection.commit()
    connection.close()


def _read_urls(path: Path) -> list[str]:
    connection = sqlite3.connect(path)
    value = connection.execute(
        "SELECT value FROM config WHERE key = ?", (CONFIG_KEY,)
    ).fetchone()[0]
    connection.close()
    return json.loads(value)


def test_connection_redirect_is_narrow_backed_up_and_reversible(tmp_path):
    database = tmp_path / "webui.db"
    backup = tmp_path / "chat-url-backup.sqlite"
    original = "http://rag-backend:8088/v1"
    capture = "http://rag-backend:8099/v1"
    _create_config_db(database, [original])

    backup_and_redirect(database, backup, original, capture)
    assert backup.exists()
    assert _read_urls(database) == [capture]

    rollback(database, backup, capture)
    assert _read_urls(database) == [original]
    assert not backup.exists()


def test_connection_redirect_refuses_multiple_or_credentialed_connections(tmp_path):
    for index, urls in enumerate(
        (["http://rag-backend:8088/v1", "http://other/v1"], ["http://user:pass@host/v1"])
    ):
        database = tmp_path / f"webui-{index}.db"
        backup = tmp_path / f"backup-{index}.sqlite"
        _create_config_db(database, urls)
        with pytest.raises(RuntimeError):
            backup_and_redirect(
                database,
                backup,
                "http://rag-backend:8088/v1",
                "http://rag-backend:8099/v1",
            )
        assert not backup.exists()
