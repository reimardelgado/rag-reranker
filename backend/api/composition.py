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
    from backend.config import GroundedImageConfig
    from backend.metrics import GroundedImageMetrics, RolloutStage, evaluate_shadow_gate
    from backend.providers.gemini_native import GeminiNativeAdapter
except ImportError:  # pragma: no cover - top-level Docker import compatibility
    from api.grounded_image import GroundedImageOrchestrator, GroundedImageRuntime
    from config import GroundedImageConfig
    from metrics import GroundedImageMetrics, RolloutStage, evaluate_shadow_gate
    from providers.gemini_native import GeminiNativeAdapter


DEFAULT_NATIVE_BASE = "https://generativelanguage.googleapis.com/v1beta"


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
        if not api_key:
            raise RuntimeError("Grounded image mode requires GEMINI_API_KEY.")
        if adapter_factory is not None:
            adapter = adapter_factory(config)
        else:
            import httpx

            client = http_client_factory() if http_client_factory else httpx.AsyncClient()
            adapter = GeminiNativeAdapter(
                api_key=api_key,
                client=client,
                base_url=environ.get("GROUNDED_IMAGE_GEMINI_BASE", DEFAULT_NATIVE_BASE),
            )

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
