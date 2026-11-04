from __future__ import annotations

import json
from pathlib import Path


FIXTURE = (
    Path(__file__).parents[1]
    / "fixtures"
    / "openwebui"
    / "v0.11.4-admin-access-decision.json"
)
RUNTIME_CAPTURE = FIXTURE.with_name("v0.11.4-runtime-capture.json")


def test_redacted_admin_export_keeps_runtime_canary_fail_closed() -> None:
    report = json.loads(FIXTURE.read_text(encoding="utf-8"))
    capture = json.loads(RUNTIME_CAPTURE.read_text(encoding="utf-8"))

    assert report["openwebui_version"] == "v0.11.4"
    assert report["export_policy"] == {
        "values_captured": False,
        "identifiers_captured": False,
        "prompts_or_evidence_captured": False,
        "credential_presence_only": True,
    }
    assert report["observed_admin_state"]["generate_image_occurrences"] == 1
    assert report["observed_admin_state"]["image_generation_calls_observed"] == 0
    assert report["observed_admin_state"]["trusted_identity_headers_forwarded"] is False
    assert report["observed_admin_state"]["trusted_service_auth_forwarded"] is False
    assert (
        report["observed_admin_state"]["global_image_generation_enabled"]
        is capture["image_configuration"]["enabled"]
    )
    assert (
        report["observed_admin_state"]["global_default_image_permission"]
        is capture["image_configuration"][
            "global_default_permission_features_image_generation"
        ]
    )
    assert (
        report["observed_admin_state"]["generate_image_occurrences"]
        == capture["task_5_1_capture"]["request"]["tools"][
            "identified_allowlisted_names"
        ].count("generate_image")
    )
    assert capture["task_5_1_capture"]["protocol_conclusion"][
        "generate_image_injected"
    ] is True
    assert (
        capture["headers"]["open_webui_to_rag_backend"]["trusted_correlation"]
        ["X-OpenWebUI-Chat-Id"]
        == "absent_by_configuration"
    )

    gates = report["required_gates"]
    assert gates["least_privilege_rbac"] == "fail"
    assert gates["native_tool_schema"] == "pass"
    assert gates["trusted_identity_delivery"] == "fail"
    assert gates["trusted_service_authentication"] == "fail"
    assert gates["backend_image_composition"] == "inactive"
    assert gates["single_replica_only"] == "pass"
    assert gates["redis_multi_replica"] == "no_go"

    decision = report["decision"]
    assert decision["status"] == "no_go"
    assert decision["authorized_canary_only"] is True
    assert decision["current_environment_enabled"] is False
    assert decision["paid_generation_approved"] is False
    assert decision["text_chat_rollback_required"] is True


def test_admin_export_has_only_allowlisted_redacted_structure() -> None:
    report = json.loads(FIXTURE.read_text(encoding="utf-8"))

    assert set(report) == {
        "schema_version",
        "openwebui_version",
        "export_policy",
        "observed_admin_state",
        "required_gates",
        "decision",
    }
    serialized = json.dumps(report, sort_keys=True).casefold()
    for forbidden in (
        "authorization",
        "api_key",
        "service_token",
        "hmac",
        "user_id",
        "chat_id",
        "message_id",
        "prompt_hash",
        "evidence_packet",
        "thought_signature",
    ):
        assert forbidden not in serialized


def test_readme_records_no_go_canary_and_rollback_without_enabling_defaults(
    project_root,
) -> None:
    readme = (project_root / "README.md").read_text(encoding="utf-8")
    env_example = (project_root / ".env.example").read_text(encoding="utf-8")
    compose = (project_root / "docker-compose.yml").read_text(encoding="utf-8")

    assert "**Decisión actual: NO-GO.**" in readme
    assert "authorized_canary_only=true" in readme
    assert "v0.11.4-admin-access-decision.json" in readme
    assert "GROUNDED_IMAGE_ENABLED=false" in readme
    assert "modo multi-réplica sigue" in readme
    assert "desactivar `GROUNDED_IMAGE_ENABLED`" in readme

    assert "GROUNDED_IMAGE_ENABLED=false" in env_example
    assert "GROUNDED_IMAGE_ENABLED=${GROUNDED_IMAGE_ENABLED:-false}" in compose
    assert "GROUNDED_IMAGE_REPLICA_COUNT=1" in compose
