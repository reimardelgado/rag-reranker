import json
import re
from pathlib import Path


FIXTURE_PATH = (
    Path(__file__).parents[1]
    / "fixtures"
    / "providers"
    / "gemini-openai-chat-conformance.json"
)

FORBIDDEN_FIELD_NAMES = {
    "api_key",
    "authorization",
    "bearer",
    "cookie",
    "password",
    "prompt",
    "raw_arguments",
    "raw_body",
    "raw_provider_text",
    "thought_signature_value",
    "token",
    "tool_call_id_value",
}

FORBIDDEN_VALUE_PATTERNS = (
    re.compile(r"(?i)bearer\s+[a-z0-9._~+/=-]+"),
    re.compile(r"(?i)(?:api[_-]?key|password|token)\s*[:=]\s*\S+"),
    re.compile(r"AIza[0-9A-Za-z_-]{20,}"),
    re.compile(r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+"),
)


def _walk(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key, child
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def test_provider_conformance_fixture_has_bounded_sanitized_schema():
    report = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    assert report["schema_version"] == 1
    assert report["task"] == "5.2"
    assert report["safe_to_commit"] is True
    assert report["scope"] == {
        "chat_endpoint_only": True,
        "custom_function_name": "capture_probe",
        "image_generation_invoked": False,
        "external_image_url_used": False,
        "rag_invoked": False,
        "raw_provider_payload_persisted": False,
    }
    assert report["call_accounting"] == {
        "total_real_calls": 10,
        "credential_health_calls": 1,
        "chat_conformance_calls": 9,
        "continuation_calls": 1,
        "retries_after_auth_or_rate_limit": 0,
    }
    assert report["credential_health"]["http_status"] == 200
    assert len(report["batches"]) == 3
    assert report["gate_result"]["task_complete"] is True
    assert report["gate_result"]["provider_decision_performed"] is False


def test_provider_conformance_fixture_records_required_gate_outcomes():
    report = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    criteria = report["direct_adapter_criteria"]

    assert set(criteria) == {
        "non_stream_tool_call",
        "stream_tool_call",
        "tool_result_continuation",
        "thought_signature",
        "multimodal_fidelity",
    }
    assert criteria["non_stream_tool_call"]["status"] == "passed"
    assert criteria["stream_tool_call"]["status"] == "not_evidenced"
    assert criteria["tool_result_continuation"]["status"] == "not_run_openai_compatible"
    assert criteria["thought_signature"]["presence_category"] == (
        "extra_content_google_thought_signature_string"
    )
    assert criteria["multimodal_fidelity"]["status"] == "failed"

    native = report["native_adapter_criteria"]
    assert native["function_call"]["status"] == "passed"
    assert native["thought_signature"]["exact_in_memory_round_trip"] is True
    assert native["function_result_continuation"]["status"] == "passed"
    assert report["feasibility"] == {
        "openai_compatible_direct_adapter": "not_fully_qualified",
        "openai_compatible_reason": (
            "stream_continuation_and_multimodal_requirements_not_evidenced"
        ),
        "native_gemini_adapter": "feasible",
        "native_reason": (
            "function_call_signature_and_exact_signature_continuation_passed"
        ),
    }


def test_provider_conformance_fixture_contains_no_sensitive_or_raw_values():
    text = FIXTURE_PATH.read_text(encoding="utf-8")
    report = json.loads(text)

    assert all(value is False for value in report["redaction"].values())
    assert not (FORBIDDEN_FIELD_NAMES & {key.lower() for key, _ in _walk(report)})
    assert all(pattern.search(text) is None for pattern in FORBIDDEN_VALUE_PATTERNS)
    assert "data:image" not in text
    assert "chatcmpl" not in text
    assert re.search(r'"(?:id|tool_call_id_value)"\s*:\s*"', text) is None
