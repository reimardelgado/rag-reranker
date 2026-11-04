"""No-forward OpenAI-compatible capture boundary for Open WebUI protocol probes.

The persisted capture is deliberately structural: JSON keys, value types, array
lengths, a very small enum allowlist, booleans, and header names. Request values
never leave process memory and the server has no forwarding code.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


MAX_BODY_BYTES = 1_048_576
MAX_DEPTH = 24
MAX_NODES = 20_000
MODEL_ID = "capture-safe-model"
SAFE_STRING_VALUES = {
    "role": {"system", "user", "assistant", "tool", "developer"},
    "name": {"generate_image"},
    "function_calling": {"native", "default", "legacy"},
    "tool_choice": {"auto", "none", "required"},
    "type": {
        "function",
        "text",
        "input_text",
        "image_url",
        "file",
    },
}
SAFE_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,127}$")


class CaptureLimitError(ValueError):
    """Raised when a body cannot be traversed within fixed safety bounds."""


def _safe_key_name(key: Any) -> str:
    text = str(key)
    return text if SAFE_KEY.fullmatch(text) else "<redacted-key>"


def structural_shape(value: Any, *, key: str | None = None) -> dict[str, Any]:
    """Return a bounded structural description with no arbitrary string values."""

    nodes = 0

    def visit(item: Any, current_key: str | None, depth: int) -> dict[str, Any]:
        nonlocal nodes
        nodes += 1
        if nodes > MAX_NODES or depth > MAX_DEPTH:
            raise CaptureLimitError("capture structure exceeds safety bounds")

        if item is None:
            return {"type": "null"}
        if isinstance(item, bool):
            return {"type": "boolean", "safe_value": item}
        if isinstance(item, str):
            result: dict[str, Any] = {"type": "string"}
            allowed = SAFE_STRING_VALUES.get(current_key or "", set())
            if item in allowed:
                result["safe_value"] = item
            return result
        if isinstance(item, int) and not isinstance(item, bool):
            return {"type": "integer"}
        if isinstance(item, float):
            return {"type": "number"}
        if isinstance(item, list):
            return {
                "type": "array",
                "length": len(item),
                "items": [visit(child, current_key, depth + 1) for child in item],
            }
        if isinstance(item, dict):
            keys: dict[str, Any] = {}
            for raw_key, child in item.items():
                safe_key = _safe_key_name(raw_key)
                # Duplicate unsafe names are intentionally coalesced: their values
                # are never evidence and retaining them could leak dynamic IDs.
                keys[safe_key] = visit(child, safe_key, depth + 1)
            return {"type": "object", "keys": keys}
        return {"type": "unsupported"}

    return visit(value, key, 0)


def header_presence(headers: Any) -> dict[str, Any]:
    """Record syntactically safe header names only, never values."""

    names = sorted({_safe_key_name(name) for name in headers.keys()}, key=str.lower)
    return {"present_names": names, "values_captured": False}


def _atomic_json_write(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


class CaptureState:
    def __init__(self, output_path: Path):
        self.output_path = output_path
        self._lock = threading.Lock()
        self.accepted_chat_requests = 0

    def capture_once(self, headers: Any, body: Any) -> bool:
        with self._lock:
            if self.accepted_chat_requests:
                return False
            capture = {
                "schema_version": 1,
                "status": "captured",
                "forwarded": False,
                "accepted_chat_requests": 1,
                "extra_chat_request_policy": "reject_409",
                "request": {
                    "method": "POST",
                    "path": "/v1/chat/completions",
                    "headers": header_presence(headers),
                    "body": structural_shape(body),
                },
                "sanitization": {
                    "arbitrary_string_values_persisted": False,
                    "header_values_persisted": False,
                    "raw_content_persisted": False,
                    "identifiers_persisted": False,
                },
            }
            _atomic_json_write(self.output_path, capture)
            self.accepted_chat_requests = 1
            return True


def make_handler(state: CaptureState):
    class RedactingCaptureHandler(BaseHTTPRequestHandler):
        server_version = "RedactingCapture/1"

        def log_message(self, _format: str, *_args: Any) -> None:
            # Default access logs can include attacker-controlled paths. Persist
            # nothing except the explicitly sanitized capture document.
            return

        def _json(self, status: int, body: dict[str, Any]) -> None:
            payload = json.dumps(body, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            if self.path in {"/v1/models", "/models"}:
                self._json(
                    200,
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": MODEL_ID,
                                "object": "model",
                                "created": 0,
                                "owned_by": "capture-fixture",
                            }
                        ],
                    },
                )
                return
            if self.path == "/health":
                self._json(200, {"status": "ok", "chat_capacity": 1})
                return
            self._json(404, {"error": {"type": "not_found"}})

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            if self.path not in {"/v1/chat/completions", "/chat/completions"}:
                self._json(404, {"error": {"type": "not_found"}})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._json(400, {"error": {"type": "invalid_request_error"}})
                return
            if length <= 0 or length > MAX_BODY_BYTES:
                self._json(413, {"error": {"type": "request_too_large"}})
                return
            try:
                body = json.loads(self.rfile.read(length))
                accepted = state.capture_once(self.headers, body)
            except (json.JSONDecodeError, UnicodeDecodeError, CaptureLimitError):
                self._json(400, {"error": {"type": "invalid_request_error"}})
                return
            if not accepted:
                self._json(409, {"error": {"type": "capture_already_complete"}})
                return

            created = 1_800_000_000
            if isinstance(body, dict) and body.get("stream") is True:
                chunks = (
                    {
                        "id": "capture-response",
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": MODEL_ID,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "role": "assistant",
                                    "content": "Temporary capture completed without forwarding.",
                                },
                                "finish_reason": None,
                            }
                        ],
                    },
                    {
                        "id": "capture-response",
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": MODEL_ID,
                        "choices": [
                            {"index": 0, "delta": {}, "finish_reason": "stop"}
                        ],
                    },
                )
                payload = b"".join(
                    f"data: {json.dumps(chunk, separators=(',', ':'))}\n\n".encode()
                    for chunk in chunks
                ) + b"data: [DONE]\n\n"
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return

            self._json(
                200,
                {
                    "id": "capture-response",
                    "object": "chat.completion",
                    "created": created,
                    "model": MODEL_ID,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "Temporary capture completed without forwarding.",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    },
                },
            )

    return RedactingCaptureHandler


def serve(host: str, port: int, output_path: Path) -> None:
    state = CaptureState(output_path)
    server = ThreadingHTTPServer((host, port), make_handler(state))
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8099)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    serve(args.host, args.port, args.output)


if __name__ == "__main__":
    main()
