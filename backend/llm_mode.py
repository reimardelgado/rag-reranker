"""Deployment-wide LLM provider mode: fully local (Ollama) or paid Gemini.

``DEFAULT_PROVIDER`` selects the mode once for the whole backend:

* ``ollama`` (default): text chat and the grounded-image agent run on the local
  Ollama server.  No Gemini key is required and no Gemini endpoint is called.
* ``gemini``: text chat uses Gemini (falling back to Ollama when Gemini fails,
  as before) and the grounded-image agent uses the native Gemini adapter.

The image *engine* (ComfyUI, AUTOMATIC1111 or Gemini) is configured in Open
WebUI, not here; see docs/grounded-image-activation.md.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


PROVIDER_OLLAMA = "ollama"
PROVIDER_GEMINI = "gemini"
PROVIDERS = frozenset({PROVIDER_OLLAMA, PROVIDER_GEMINI})

# Small enough for a 4 GB GPU (or CPU); production overrides it in .env.
DEFAULT_OLLAMA_MODEL = "qwen2.5:3b"
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"


class LLMModeError(ValueError):
    """A safe configuration failure; values are never echoed."""


def is_gemini_model(model_id: str) -> bool:
    return model_id.startswith(("gemini-", "models/gemini"))


def _model_name(environ: Mapping[str, str], name: str, default: str) -> str:
    value = environ.get(name)
    if value is None or value == "":
        return default
    if value != value.strip() or any(ord(c) < 0x21 or ord(c) == 0x7F for c in value):
        raise LLMModeError(f"{name} is malformed.")
    if len(value.encode("utf-8")) > 256:
        raise LLMModeError(f"{name} is too long.")
    return value


def provider_from_environ(environ: Mapping[str, str]) -> str:
    value = environ.get("DEFAULT_PROVIDER")
    if value is None or value == "":
        return PROVIDER_OLLAMA
    normalized = value.strip().lower()
    if normalized not in PROVIDERS:
        raise LLMModeError("DEFAULT_PROVIDER must be 'ollama' or 'gemini'.")
    return normalized


@dataclass(frozen=True, slots=True)
class LLMMode:
    provider: str
    ollama_model: str
    gemini_model: str

    @classmethod
    def from_environ(cls, environ: Mapping[str, str]) -> "LLMMode":
        return cls(
            provider=provider_from_environ(environ),
            ollama_model=_model_name(environ, "OLLAMA_DEFAULT_MODEL", DEFAULT_OLLAMA_MODEL),
            gemini_model=_model_name(environ, "DEFAULT_GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
        )

    @property
    def is_local(self) -> bool:
        return self.provider == PROVIDER_OLLAMA

    @property
    def requires_gemini_key(self) -> bool:
        return not self.is_local

    @property
    def default_chat_model(self) -> str:
        return self.ollama_model if self.is_local else self.gemini_model

    def routes_to_ollama(self, model_id: str) -> bool:
        """Local mode never reaches Gemini, whatever model name a client sends."""

        return self.is_local or not is_gemini_model(model_id)

    def resolve_local_model(self, model_id: str) -> str:
        """Map a Gemini model name to the local default in local mode."""

        if self.is_local and is_gemini_model(model_id):
            return self.ollama_model
        return model_id
