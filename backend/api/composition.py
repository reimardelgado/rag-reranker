"""Flag-driven composition of the grounded image runtime.

``build_grounded_image_composition`` returns ``(None, None)`` when both
``GROUNDED_IMAGE_ENABLED`` and ``GROUNDED_IMAGE_SHADOW`` are false, so the
default deployment is byte-for-byte the established text-only service.
Shadow mode composes only intent/retrieval/metrics, with no provider adapter
and therefore no possible provider or image-engine call.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from fastapi import APIRouter

try:
    from backend.api.grounded_image import GroundedImageOrchestrator, GroundedImageRuntime
    from backend.config import ADAPTER_OLLAMA, ConfigurationError, GroundedImageConfig
    from backend.metrics import GroundedImageMetrics, RolloutStage, evaluate_shadow_gate
    from backend.providers.gemini_native import GeminiNativeAdapter
    from backend.providers.ollama_native import OllamaNativeAdapter
except ImportError:  # pragma: no cover - top-level Docker import compatibility
    from api.grounded_image import GroundedImageOrchestrator, GroundedImageRuntime
    from config import ADAPTER_OLLAMA, ConfigurationError, GroundedImageConfig
    from metrics import GroundedImageMetrics, RolloutStage, evaluate_shadow_gate
    from providers.gemini_native import GeminiNativeAdapter
    from providers.ollama_native import OllamaNativeAdapter


DEFAULT_NATIVE_BASE = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_OLLAMA_BASE = "http://ollama:11434"
# Local generation on a 4 GB GPU (or CPU) can take minutes per round.
DEFAULT_OLLAMA_TIMEOUT_SECONDS = 300


def _ollama_timeout(environ: Mapping[str, str]) -> float:
    value = environ.get("OLLAMA_TIMEOUT_SECONDS")
    if value is None or value == "":
        return float(DEFAULT_OLLAMA_TIMEOUT_SECONDS)
    if not value.isascii() or not value.isdecimal() or not 10 <= int(value) <= 3600:
        raise ConfigurationError("OLLAMA_TIMEOUT_SECONDS must be an integer in [10, 3600].")
    return float(value)


def _default_adapter(
    config: GroundedImageConfig,
    environ: Mapping[str, str],
    api_key: str | None,
    http_client_factory: Callable[[], object] | None,
) -> object:
    import httpx

    client = http_client_factory() if http_client_factory else httpx.AsyncClient()
    if config.adapter == ADAPTER_OLLAMA:
        return OllamaNativeAdapter(
            client=client,
            base_url=environ.get("OLLAMA_BASE_URL") or DEFAULT_OLLAMA_BASE,
            timeout=httpx.Timeout(_ollama_timeout(environ), connect=5.0),
            keep_alive=environ.get("OLLAMA_KEEP_ALIVE") or None,
        )
    if not api_key:
        raise RuntimeError("Grounded image mode with gemini_native requires GEMINI_API_KEY.")
    return GeminiNativeAdapter(
        api_key=api_key,
        client=client,
        base_url=environ.get("GROUNDED_IMAGE_GEMINI_BASE", DEFAULT_NATIVE_BASE),
    )


def _metrics_router(orchestrator: GroundedImageOrchestrator) -> APIRouter:
    router = APIRouter()

    @router.get("/v1/grounded-image/metrics", include_in_schema=False)
    async def grounded_image_metrics():
        """Aggregate-only snapshot plus the conservative shadow-gate verdict."""

        snapshot = orchestrator.metrics.snapshot()
        decision = evaluate_shadow_gate(snapshot)
        return {
            "snapshot": snapshot,
            "shadow_gate": {
                "outcome": decision.outcome,
                "blockers": list(decision.blockers),
            },
            # Counts of requests carrying each trusted header; never values.
            "trusted_header_presence": orchestrator.header_presence(),
        }

    return router


def build_grounded_image_composition(
    environ: Mapping[str, str],
    *,
    retrieve_blocks: Callable[[str, int], list],
    retrieval_model_id: str,
    api_key: str | None,
    http_client_factory: Callable[[], object] | None = None,
    adapter_factory: Callable[..., object] | None = None,
    top_k: int = 8,
) -> tuple[GroundedImageOrchestrator | None, APIRouter | None]:
    config = GroundedImageConfig.from_environ(environ)
    if not (config.enabled or config.shadow):
        return None, None

    adapter = None
    if config.enabled:
        if adapter_factory is not None:
            adapter = adapter_factory(config)
        else:
            adapter = _default_adapter(config, environ, api_key, http_client_factory)

    metrics = GroundedImageMetrics(
        RolloutStage.CANARY if config.enabled else RolloutStage.SHADOW
    )
    orchestrator = GroundedImageOrchestrator(
        GroundedImageRuntime(
            config=config,
            adapter=adapter,
            retrieve_blocks=retrieve_blocks,
            retrieval_model_id=retrieval_model_id,
            metrics=metrics,
            top_k=top_k,
        )
    )
    return orchestrator, _metrics_router(orchestrator)
