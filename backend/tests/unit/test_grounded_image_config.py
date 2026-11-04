from __future__ import annotations

import base64
from copy import deepcopy

import pytest

from backend.config import (
    ConfigurationError,
    GroundedImageAccessGate,
    GroundedImageConfig,
    ImageAccessDenied,
)


SERVICE_TOKEN = "service-token-" + "s" * 32
HMAC_KEY = base64.urlsafe_b64encode(b"h" * 32).decode().rstrip("=")


def enabled_environ() -> dict[str, str]:
    return {
        "GROUNDED_IMAGE_ENABLED": "true",
        "GROUNDED_IMAGE_TRUSTED_BOUNDARY": "true",
        "GROUNDED_IMAGE_PROVENANCE_ENABLED": "true",
        "GROUNDED_IMAGE_AUTHORIZED_USERS": "test-user-a,test-user-b",
        "GROUNDED_IMAGE_AUTHORIZED_MODELS": "test-model-a,test-model-b",
        "RAG_TRUSTED_SERVICE_TOKEN": SERVICE_TOKEN,
        "RAG_CORRELATION_HMAC_KEY": HMAC_KEY,
    }


def trusted_headers(*, user: str = "test-user-a", token: str = SERVICE_TOKEN):
    return {
        "X-OpenWebUI-User-Id": user,
        "X-OpenWebUI-Chat-Id": "test-chat",
        "X-OpenWebUI-Message-Id": "test-root",
        "X-RAG-Service-Token": token,
    }


def test_defaults_are_disabled_empty_single_replica_and_normatively_bounded() -> None:
    config = GroundedImageConfig.from_environ({})

    assert config.enabled is False
    assert config.trusted_boundary is False
    assert config.provenance_enabled is False
    assert config.authorized_users == frozenset()
    assert config.authorized_models == frozenset()
    assert config.service_token is None
    assert config.correlation_hmac_key is None
    assert config.minimum_references == 2
    assert config.max_tool_iterations == 3
    assert config.replica_count == 1
    assert config.correlation_limits().ttl_seconds == 600
    assert config.correlation_limits().max_packet_bytes == 256 * 1024


def test_disabled_gate_denies_before_reading_headers() -> None:
    class UnreadableHeaders(dict):
        def items(self):
            raise AssertionError("disabled access inspected trusted headers")

    gate = GroundedImageAccessGate(GroundedImageConfig.from_environ({}))
    with pytest.raises(ImageAccessDenied, match="IMAGE_DISABLED"):
        gate.authorize(
            UnreadableHeaders(),
            model_id="anything",
            permission_granted=True,
            capability_enabled=True,
            image_toggle_enabled=True,
        )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("GROUNDED_IMAGE_ENABLED", "TRUE"),
        ("GROUNDED_IMAGE_MIN_REFERENCES", "1"),
        ("GROUNDED_IMAGE_CORRELATION_TTL_SECONDS", "601"),
        ("GROUNDED_IMAGE_CORRELATION_MAX_PACKET_BYTES", "262145"),
        ("GROUNDED_IMAGE_MAX_TOOL_ITERATIONS", "4"),
        ("GROUNDED_IMAGE_AUTHORIZED_USERS", "user-a, user-b"),
        ("RAG_TRUSTED_SERVICE_TOKEN", "short"),
        ("RAG_CORRELATION_HMAC_KEY", "not base64!"),
    ],
)
def test_explicit_malformed_or_unsafe_values_are_rejected_even_while_disabled(
    name: str, value: str
) -> None:
    with pytest.raises(ConfigurationError):
        GroundedImageConfig.from_environ({name: value})


@pytest.mark.parametrize(
    "remove_or_replace",
    [
        lambda values: values.__setitem__("GROUNDED_IMAGE_TRUSTED_BOUNDARY", "false"),
        lambda values: values.__setitem__("GROUNDED_IMAGE_PROVENANCE_ENABLED", "false"),
        lambda values: values.__setitem__("GROUNDED_IMAGE_AUTHORIZED_USERS", ""),
        lambda values: values.__setitem__("GROUNDED_IMAGE_AUTHORIZED_MODELS", ""),
        lambda values: values.pop("RAG_TRUSTED_SERVICE_TOKEN"),
        lambda values: values.pop("RAG_CORRELATION_HMAC_KEY"),
        lambda values: values.__setitem__("GROUNDED_IMAGE_ADAPTER", "direct_openai"),
        lambda values: values.__setitem__("GROUNDED_IMAGE_REPLICA_COUNT", "2"),
    ],
)
def test_activation_requires_every_security_and_single_replica_gate(
    remove_or_replace,
) -> None:
    values = enabled_environ()
    remove_or_replace(values)
    with pytest.raises(ConfigurationError):
        GroundedImageConfig.from_environ(values)


def test_enabled_config_is_detached_and_redacts_secrets_and_allowlists() -> None:
    values = enabled_environ()
    original = deepcopy(values)

    config = GroundedImageConfig.from_environ(values)
    assert values == original
    values["GROUNDED_IMAGE_AUTHORIZED_USERS"] = "mutated"
    rendered = repr(config)

    assert config.authorized_users == frozenset({"test-user-a", "test-user-b"})
    for private_value in (
        SERVICE_TOKEN,
        HMAC_KEY,
        "test-user-a",
        "test-user-b",
        "test-model-a",
        "test-model-b",
    ):
        assert private_value not in rendered


def test_access_requires_openwebui_permission_capability_toggle_and_model() -> None:
    gate = GroundedImageAccessGate(
        GroundedImageConfig.from_environ(enabled_environ())
    )
    cases = [
        {"permission_granted": False, "capability_enabled": True, "image_toggle_enabled": True},
        {"permission_granted": True, "capability_enabled": False, "image_toggle_enabled": True},
        {"permission_granted": True, "capability_enabled": True, "image_toggle_enabled": False},
    ]
    for flags in cases:
        with pytest.raises(ImageAccessDenied, match="FORBIDDEN"):
            gate.authorize(trusted_headers(), model_id="test-model-a", **flags)

    with pytest.raises(ImageAccessDenied, match="FORBIDDEN"):
        gate.authorize(
            trusted_headers(),
            model_id="unlisted-model",
            permission_granted=True,
            capability_enabled=True,
            image_toggle_enabled=True,
        )


def test_access_authenticates_trusted_headers_then_enforces_user_allowlist() -> None:
    config = GroundedImageConfig.from_environ(enabled_environ())
    gate = GroundedImageAccessGate(config)
    flags = {
        "model_id": "test-model-a",
        "permission_granted": True,
        "capability_enabled": True,
        "image_toggle_enabled": True,
    }

    headers = trusted_headers()
    original_headers = deepcopy(headers)
    grant = gate.authorize(headers, **flags)
    assert headers == original_headers
    assert len(grant.correlation_key.digest) == 64
    assert "test-user-a" not in repr(grant)

    with pytest.raises(ImageAccessDenied, match="AUTH_REQUIRED"):
        gate.authorize(trusted_headers(token="x" * 40), **flags)
    with pytest.raises(ImageAccessDenied, match="FORBIDDEN"):
        gate.authorize(trusted_headers(user="unlisted-user"), **flags)


def test_documented_environment_and_compose_defaults_remain_off(project_root) -> None:
    env_text = (project_root / ".env.example").read_text(encoding="utf-8")
    compose_text = (project_root / "docker-compose.yml").read_text(encoding="utf-8")

    assert "GROUNDED_IMAGE_ENABLED=false" in env_text
    assert "RAG_TRUSTED_SERVICE_TOKEN=\n" in env_text
    assert "RAG_CORRELATION_HMAC_KEY=\n" in env_text
    assert "GROUNDED_IMAGE_ENABLED=${GROUNDED_IMAGE_ENABLED:-false}" in compose_text
    assert "GROUNDED_IMAGE_REPLICA_COUNT=1" in compose_text
