"""Application configuration and inactive grounded-image access gates.

Grounded image execution remains opt-in.  Configuration is parsed strictly so
an ambiguous or incomplete deployment cannot accidentally enable a paid path.
Secrets and trusted identifiers are deliberately excluded from representations.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping
from dataclasses import dataclass, field
import os
import re
import secrets

from dotenv import load_dotenv

try:
    from backend.correlation import CorrelationKey, CorrelationLimits, TrustedHeaderCorrelator
    from backend.llm_mode import LLMMode, LLMModeError
except ImportError:  # Docker and maintenance scripts import top-level ``config``.
    from correlation import CorrelationKey, CorrelationLimits, TrustedHeaderCorrelator
    from llm_mode import LLMMode, LLMModeError


load_dotenv()


class ConfigurationError(ValueError):
    """A safe configuration failure; values are never included in the message."""


class ImageAccessDenied(PermissionError):
    """Fail-closed image access denial with a stable public error code."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _strict_bool(environ: Mapping[str, str], name: str, default: bool) -> bool:
    value = environ.get(name)
    if value is None:
        return default
    if not isinstance(value, str) or value not in {"true", "false"}:
        raise ConfigurationError(f"{name} must be exactly 'true' or 'false'.")
    return value == "true"


def _bounded_int(
    environ: Mapping[str, str],
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    value = environ.get(name)
    if value is None:
        return default
    if not isinstance(value, str) or not value.isascii() or not value.isdecimal():
        raise ConfigurationError(f"{name} must be a decimal integer.")
    parsed = int(value)
    if not minimum <= parsed <= maximum:
        raise ConfigurationError(f"{name} is outside its permitted bound.")
    return parsed


_FRACTION = re.compile(r"^(0\.[0-9]{1,4}|1(\.0{1,4})?)$")


def _bounded_fraction(
    environ: Mapping[str, str], name: str, default: float
) -> float:
    """Parse a strict decimal in (0, 1], e.g. ``0.6`` or ``1``."""

    value = environ.get(name)
    if value is None:
        return default
    if not isinstance(value, str) or not _FRACTION.match(value):
        raise ConfigurationError(f"{name} must be a decimal in (0, 1].")
    parsed = float(value)
    if parsed <= 0:
        raise ConfigurationError(f"{name} is outside its permitted bound.")
    return parsed


def _allowlist(
    environ: Mapping[str, str], name: str, *, max_item_bytes: int
) -> frozenset[str]:
    value = environ.get(name)
    if value is None or value == "":
        return frozenset()
    if not isinstance(value, str):
        raise ConfigurationError(f"{name} must be a comma-separated string.")
    items = value.split(",")
    if any(
        not item
        or item != item.strip()
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in item)
        or len(item.encode("utf-8")) > max_item_bytes
        for item in items
    ):
        raise ConfigurationError(f"{name} contains an invalid entry.")
    if len(items) != len(set(items)):
        raise ConfigurationError(f"{name} contains duplicate entries.")
    return frozenset(items)


def _bounded_text(
    environ: Mapping[str, str], name: str, default: str, *, max_bytes: int = 256
) -> str:
    value = environ.get(name)
    if value is None or value == "":
        return default
    if (
        not isinstance(value, str)
        or value != value.strip()
        or any(ord(character) < 0x21 or ord(character) == 0x7F for character in value)
        or len(value.encode("utf-8")) > max_bytes
    ):
        raise ConfigurationError(f"{name} is malformed.")
    return value


CORRELATION_SCOPES = frozenset({"message", "chat"})
ADAPTER_GEMINI = "gemini_native"
ADAPTER_OLLAMA = "ollama"
IMAGE_ADAPTERS = frozenset({ADAPTER_GEMINI, ADAPTER_OLLAMA})
PROMPT_FORMATS = frozenset({"canonical_json", "diffusion"})


def _secret(environ: Mapping[str, str], name: str) -> str | None:
    value = environ.get(name)
    if value is None or value == "":
        return None
    if (
        not isinstance(value, str)
        or value != value.strip()
        or any(ord(character) < 0x21 or ord(character) == 0x7F for character in value)
        or not 32 <= len(value.encode("utf-8")) <= 512
    ):
        raise ConfigurationError(f"{name} is malformed.")
    return value


def _hmac_key(environ: Mapping[str, str], name: str) -> bytes | None:
    value = environ.get(name)
    if value is None or value == "":
        return None
    if not isinstance(value, str) or value != value.strip():
        raise ConfigurationError(f"{name} is malformed.")
    try:
        padded = value + "=" * (-len(value) % 4)
        decoded = base64.b64decode(padded, altchars=b"-_", validate=True)
    except (ValueError, TypeError):
        raise ConfigurationError(f"{name} must be base64url encoded.") from None
    if not 32 <= len(decoded) <= 128:
        raise ConfigurationError(f"{name} must decode to 32-128 bytes.")
    return decoded


@dataclass(frozen=True, slots=True)
class GroundedImageConfig:
    """Strict default-deny configuration for the future image composition root."""

    enabled: bool = False
    shadow: bool = False
    trusted_boundary: bool = False
    provenance_enabled: bool = False
    adapter: str = "gemini_native"
    authorized_users: frozenset[str] = field(default_factory=frozenset, repr=False)
    authorized_models: frozenset[str] = field(default_factory=frozenset, repr=False)
    service_token: str | None = field(default=None, repr=False)
    correlation_hmac_key: bytes | None = field(default=None, repr=False)
    minimum_references: int = 2
    min_subject_coverage: float = 0.5
    correlation_ttl_seconds: int = 600
    correlation_max_entries: int = 1024
    correlation_max_packet_bytes: int = 256 * 1024
    correlation_max_identifier_bytes: int = 1024
    max_tool_iterations: int = 3
    replica_count: int = 1
    chat_model: str = "gemini-3.8-flash"
    image_model_id: str = "open-webui-image-engine"
    correlation_scope: str = "message"
    # canonical_json for language-model engines (Gemini); diffusion for
    # CLIP-based engines (ComfyUI/A1111). Defaults follow DEFAULT_PROVIDER.
    prompt_format: str = "canonical_json"

    def __post_init__(self) -> None:
        self._validate_activation()

    @classmethod
    def from_environ(cls, environ: Mapping[str, str]) -> "GroundedImageConfig":
        """Parse a detached configuration snapshot without mutating ``environ``.

        ``DEFAULT_PROVIDER`` supplies the defaults for the adapter and the
        agent model: local mode uses Ollama, gemini mode the native adapter.
        Explicit ``GROUNDED_IMAGE_ADAPTER``/``GROUNDED_IMAGE_CHAT_MODEL`` win.
        """

        try:
            mode = LLMMode.from_environ(environ)
        except LLMModeError as error:
            raise ConfigurationError(str(error)) from None
        default_adapter = ADAPTER_OLLAMA if mode.is_local else ADAPTER_GEMINI
        config = cls(
            enabled=_strict_bool(environ, "GROUNDED_IMAGE_ENABLED", False),
            shadow=_strict_bool(environ, "GROUNDED_IMAGE_SHADOW", False),
            trusted_boundary=_strict_bool(
                environ, "GROUNDED_IMAGE_TRUSTED_BOUNDARY", False
            ),
            provenance_enabled=_strict_bool(
                environ, "GROUNDED_IMAGE_PROVENANCE_ENABLED", False
            ),
            adapter=environ.get("GROUNDED_IMAGE_ADAPTER") or default_adapter,
            authorized_users=_allowlist(
                environ, "GROUNDED_IMAGE_AUTHORIZED_USERS", max_item_bytes=1024
            ),
            authorized_models=_allowlist(
                environ, "GROUNDED_IMAGE_AUTHORIZED_MODELS", max_item_bytes=256
            ),
            service_token=_secret(environ, "RAG_TRUSTED_SERVICE_TOKEN"),
            correlation_hmac_key=_hmac_key(environ, "RAG_CORRELATION_HMAC_KEY"),
            minimum_references=_bounded_int(
                environ,
                "GROUNDED_IMAGE_MIN_REFERENCES",
                2,
                minimum=2,
                maximum=128,
            ),
            min_subject_coverage=_bounded_fraction(
                environ, "GROUNDED_IMAGE_MIN_SUBJECT_COVERAGE", 0.5
            ),
            correlation_ttl_seconds=_bounded_int(
                environ,
                "GROUNDED_IMAGE_CORRELATION_TTL_SECONDS",
                600,
                minimum=600,
                maximum=600,
            ),
            correlation_max_entries=_bounded_int(
                environ,
                "GROUNDED_IMAGE_CORRELATION_MAX_ENTRIES",
                1024,
                minimum=1,
                maximum=100_000,
            ),
            correlation_max_packet_bytes=_bounded_int(
                environ,
                "GROUNDED_IMAGE_CORRELATION_MAX_PACKET_BYTES",
                256 * 1024,
                minimum=256 * 1024,
                maximum=256 * 1024,
            ),
            correlation_max_identifier_bytes=_bounded_int(
                environ,
                "GROUNDED_IMAGE_CORRELATION_MAX_IDENTIFIER_BYTES",
                1024,
                minimum=1,
                maximum=1024,
            ),
            max_tool_iterations=_bounded_int(
                environ,
                "GROUNDED_IMAGE_MAX_TOOL_ITERATIONS",
                3,
                minimum=3,
                maximum=3,
            ),
            replica_count=_bounded_int(
                environ,
                "GROUNDED_IMAGE_REPLICA_COUNT",
                1,
                minimum=1,
                maximum=100_000,
            ),
            chat_model=_bounded_text(
                environ, "GROUNDED_IMAGE_CHAT_MODEL", mode.default_chat_model
            ),
            image_model_id=_bounded_text(
                environ, "GROUNDED_IMAGE_IMAGE_MODEL_ID", "open-webui-image-engine"
            ),
            correlation_scope=_bounded_text(
                environ, "GROUNDED_IMAGE_CORRELATION_SCOPE", "message", max_bytes=16
            ),
            prompt_format=_bounded_text(
                environ,
                "GROUNDED_IMAGE_PROMPT_FORMAT",
                "diffusion" if mode.is_local else "canonical_json",
                max_bytes=32,
            ),
        )
        return config

    def _validate_activation(self) -> None:
        if not isinstance(self.adapter, str) or self.adapter not in IMAGE_ADAPTERS:
            raise ConfigurationError("GROUNDED_IMAGE_ADAPTER is unsupported.")
        if self.prompt_format not in PROMPT_FORMATS:
            raise ConfigurationError("GROUNDED_IMAGE_PROMPT_FORMAT is unsupported.")
        if self.correlation_scope not in CORRELATION_SCOPES:
            raise ConfigurationError("GROUNDED_IMAGE_CORRELATION_SCOPE is unsupported.")
        if self.enabled and self.shadow:
            raise ConfigurationError(
                "Image mode and shadow mode are mutually exclusive."
            )
        if not self.enabled:
            return
        if not self.trusted_boundary:
            raise ConfigurationError("Image mode requires a trusted private boundary.")
        if not self.provenance_enabled:
            raise ConfigurationError("Image mode requires adjacent provenance.")
        if not self.authorized_users or not self.authorized_models:
            raise ConfigurationError("Image mode requires non-empty access allowlists.")
        if self.service_token is None or self.correlation_hmac_key is None:
            raise ConfigurationError("Image mode requires trusted correlation credentials.")
        if self.replica_count != 1:
            raise ConfigurationError(
                "Image mode is limited to one replica until the Redis gate passes."
            )

    def correlation_limits(self) -> CorrelationLimits:
        return CorrelationLimits(
            ttl_seconds=self.correlation_ttl_seconds,
            max_entries=self.correlation_max_entries,
            max_packet_bytes=self.correlation_max_packet_bytes,
            max_identifier_bytes=self.correlation_max_identifier_bytes,
        )


@dataclass(frozen=True, slots=True)
class ImageAccessGrant:
    """Opaque successful authorization; no raw trusted identity is retained."""

    correlation_key: CorrelationKey


class GroundedImageAccessGate:
    """Authorize cheaply and fail closed before retrieval or provider execution."""

    def __init__(self, config: GroundedImageConfig) -> None:
        self._config = config

    def authorize(
        self,
        headers: Mapping[str, str],
        *,
        model_id: str,
        permission_granted: bool,
        capability_enabled: bool,
        image_toggle_enabled: bool,
    ) -> ImageAccessGrant:
        if not self._config.enabled:
            raise ImageAccessDenied("IMAGE_DISABLED")
        if not all(
            value is True
            for value in (permission_granted, capability_enabled, image_toggle_enabled)
        ):
            raise ImageAccessDenied("FORBIDDEN")
        if not isinstance(model_id, str) or model_id not in self._config.authorized_models:
            raise ImageAccessDenied("FORBIDDEN")

        # Activation validation guarantees these values exist.  Keep the checks
        # explicit so a manually constructed invalid object still fails closed.
        if self._config.service_token is None or self._config.correlation_hmac_key is None:
            raise ImageAccessDenied("AUTH_REQUIRED")
        correlator = TrustedHeaderCorrelator(
            hmac_key=self._config.correlation_hmac_key,
            service_token=self._config.service_token,
            max_identifier_bytes=self._config.correlation_max_identifier_bytes,
        )
        try:
            correlation_key = correlator.derive_key(headers)
        except ValueError:
            raise ImageAccessDenied("AUTH_REQUIRED") from None

        normalized = {name.casefold(): value for name, value in headers.items()}
        user_id = normalized.get("x-openwebui-user-id")
        if not isinstance(user_id, str) or not any(
            secrets.compare_digest(user_id.encode("utf-8"), allowed.encode("utf-8"))
            for allowed in self._config.authorized_users
        ):
            raise ImageAccessDenied("FORBIDDEN")
        return ImageAccessGrant(correlation_key=correlation_key)


class Settings:
    """Legacy settings plus the inactive grounded-image configuration snapshot."""

    def __init__(self, environ: Mapping[str, str] | None = None) -> None:
        source = os.environ if environ is None else environ
        self.CHROMA_HOST = source.get("CHROMA_HOST", "chroma")
        self.CHROMA_PORT = int(source.get("CHROMA_PORT", 8000))
        self.COLLECTION = source.get("COLLECTION", "kb_docs")
        self.EMBED_MODEL = source.get(
            "EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.GEMINI_API_KEY = source.get("GEMINI_API_KEY")
        self.GEM_BASE = source.get(
            "GEM_BASE", "https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        self.DOCS_PATH = source.get("DOCS_PATH", "/app/docs")
        self.DEFAULT_TOP_K = int(source.get("DEFAULT_TOP_K", 6))
        self.DEFAULT_GEMINI_MODEL = source.get(
            "DEFAULT_GEMINI_MODEL", "gemini-3.8-flash"
        )
        self.GROUNDED_IMAGE = GroundedImageConfig.from_environ(source)


settings = Settings()
