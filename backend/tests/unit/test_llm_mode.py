import pytest

from backend.api.composition import build_grounded_image_composition
from backend.config import ConfigurationError, GroundedImageConfig
from backend.llm_mode import DEFAULT_OLLAMA_MODEL, LLMMode, LLMModeError
from backend.providers.gemini_native import GeminiNativeAdapter
from backend.providers.ollama_native import OllamaNativeAdapter


def test_local_mode_is_the_default_and_needs_no_gemini_key():
    mode = LLMMode.from_environ({})
    assert mode.is_local and not mode.requires_gemini_key
    assert mode.default_chat_model == DEFAULT_OLLAMA_MODEL


def test_local_mode_never_routes_to_gemini_and_maps_gemini_names():
    mode = LLMMode.from_environ({"DEFAULT_PROVIDER": "ollama", "OLLAMA_DEFAULT_MODEL": "qwen2.5:14b"})
    assert mode.routes_to_ollama("gemini-3.8-flash")
    assert mode.resolve_local_model("gemini-3.8-flash") == "qwen2.5:14b"
    assert mode.resolve_local_model("llama3.1:8b") == "llama3.1:8b"


def test_gemini_mode_keeps_legacy_routing():
    mode = LLMMode.from_environ({"DEFAULT_PROVIDER": "Gemini"})
    assert mode.requires_gemini_key
    assert mode.default_chat_model == "gemini-3.8-flash"
    assert not mode.routes_to_ollama("gemini-3.8-flash")
    assert mode.routes_to_ollama("qwen2.5:3b")
    assert mode.resolve_local_model("gemini-3.8-flash") == "gemini-3.8-flash"


@pytest.mark.parametrize("env", [{"DEFAULT_PROVIDER": "openai"}, {"OLLAMA_DEFAULT_MODEL": "bad model"}])
def test_invalid_mode_configuration_is_rejected(env):
    with pytest.raises(LLMModeError):
        LLMMode.from_environ(env)


def test_image_config_defaults_follow_the_mode():
    local = GroundedImageConfig.from_environ({"OLLAMA_DEFAULT_MODEL": "qwen2.5:14b"})
    assert local.adapter == "ollama" and local.chat_model == "qwen2.5:14b"
    paid = GroundedImageConfig.from_environ({"DEFAULT_PROVIDER": "gemini"})
    assert paid.adapter == "gemini_native" and paid.chat_model == "gemini-3.8-flash"
    explicit = GroundedImageConfig.from_environ(
        {"DEFAULT_PROVIDER": "ollama", "GROUNDED_IMAGE_ADAPTER": "gemini_native"}
    )
    assert explicit.adapter == "gemini_native"
    with pytest.raises(ConfigurationError):
        GroundedImageConfig.from_environ({"DEFAULT_PROVIDER": "nope"})


ENABLED = {
    "GROUNDED_IMAGE_ENABLED": "true",
    "GROUNDED_IMAGE_TRUSTED_BOUNDARY": "true",
    "GROUNDED_IMAGE_PROVENANCE_ENABLED": "true",
    "GROUNDED_IMAGE_AUTHORIZED_USERS": "u1",
    "GROUNDED_IMAGE_AUTHORIZED_MODELS": "m1",
    "RAG_TRUSTED_SERVICE_TOKEN": "s" * 40,
    "RAG_CORRELATION_HMAC_KEY": "a" * 43,
}


def _adapter_of(orchestrator):
    return orchestrator._runtime.adapter


def test_enabled_local_composition_builds_ollama_adapter_without_key():
    env = {**ENABLED, "OLLAMA_BASE_URL": "http://host.docker.internal:11434", "OLLAMA_TIMEOUT_SECONDS": "600"}
    orchestrator, _ = build_grounded_image_composition(
        env, retrieve_blocks=lambda q, k: [], retrieval_model_id="emb", api_key=None
    )
    adapter = _adapter_of(orchestrator)
    assert isinstance(adapter, OllamaNativeAdapter)
    assert "host.docker.internal" in repr(adapter)


def test_enabled_gemini_composition_still_requires_key():
    env = {**ENABLED, "DEFAULT_PROVIDER": "gemini"}
    with pytest.raises(RuntimeError):
        build_grounded_image_composition(
            env, retrieve_blocks=lambda q, k: [], retrieval_model_id="emb", api_key=None
        )
    orchestrator, _ = build_grounded_image_composition(
        env, retrieve_blocks=lambda q, k: [], retrieval_model_id="emb", api_key="key"
    )
    assert isinstance(_adapter_of(orchestrator), GeminiNativeAdapter)


def test_invalid_ollama_timeout_is_rejected():
    with pytest.raises(ConfigurationError):
        build_grounded_image_composition(
            {**ENABLED, "OLLAMA_TIMEOUT_SECONDS": "5"},
            retrieve_blocks=lambda q, k: [],
            retrieval_model_id="emb",
            api_key=None,
        )
