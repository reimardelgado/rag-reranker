from __future__ import annotations

import json

import pytest

from backend.metrics import (
    CounterMetric,
    GroundedImageMetrics,
    LatencyMetric,
    MetricsContractError,
    MIN_CANARY_IMAGE_RESULTS,
    MIN_CANARY_TEXT_REQUESTS,
    MIN_SHADOW_IMAGE_INTENTS,
    MIN_SHADOW_TEXT_REQUESTS,
    RolloutEvidence,
    RolloutStage,
    evaluate_canary_gate,
    evaluate_shadow_gate,
)


def test_snapshot_is_fixed_cardinality_detached_and_redacted() -> None:
    metrics = GroundedImageMetrics(RolloutStage.SHADOW)
    metrics.increment(CounterMetric.REQUESTS, "image_intent")
    metrics.increment(CounterMetric.DECISIONS, "refused", amount=2)
    metrics.observe_latency(LatencyMetric.WARM_GROUNDING, 81.5)

    snapshot = metrics.snapshot()
    snapshot["stage"] = "mutated"
    rendered = json.dumps(metrics.snapshot(), sort_keys=True)

    assert metrics.snapshot()["stage"] == "shadow"
    assert "fixed-cardinality-aggregates-only" in rendered
    for forbidden in (
        "authorization",
        "service_token",
        "hmac_key",
        "user_id",
        "chat_id",
        "message_id",
        "prompt",
        "evidence",
        "thought_signature",
    ):
        assert forbidden not in rendered.casefold()


@pytest.mark.parametrize(
    ("call",),
    [
        (lambda recorder: recorder.increment(CounterMetric.REQUESTS, "raw-user-id"),),
        (lambda recorder: recorder.increment(CounterMetric.REQUESTS, "text", amount=0),),
        (lambda recorder: recorder.observe_latency(LatencyMetric.RETRIEVAL, -1),),
        (lambda recorder: recorder.observe_latency(LatencyMetric.RETRIEVAL, float("inf")),),
        (lambda recorder: recorder.observe_latency(LatencyMetric.RETRIEVAL, 60_001),),
    ],
)
def test_observations_reject_arbitrary_labels_and_unbounded_values(call) -> None:
    with pytest.raises(MetricsContractError):
        call(GroundedImageMetrics())


def test_histogram_reports_conservative_p95_bucket_without_raw_samples() -> None:
    metrics = GroundedImageMetrics(RolloutStage.CANARY)
    for value in (10, 20, 50, 90, 101):
        metrics.observe_latency(LatencyMetric.CACHED_CONTINUATION, value)

    histogram = metrics.snapshot()["latencies"][LatencyMetric.CACHED_CONTINUATION.value]
    assert histogram["count"] == 5
    assert histogram["p95_upper_bound_ms"] == 200
    assert set(histogram) == {
        "count",
        "sum_ms",
        "max_ms",
        "p95_upper_bound_ms",
        "buckets",
    }


def test_shadow_gate_requires_samples_and_forbids_provider_or_paid_activity() -> None:
    metrics = GroundedImageMetrics(RolloutStage.SHADOW)
    metrics.increment(CounterMetric.SHADOW, "evaluated", amount=MIN_SHADOW_IMAGE_INTENTS)
    metrics.increment(CounterMetric.REQUESTS, "text", amount=MIN_SHADOW_TEXT_REQUESTS)
    metrics.increment(CounterMetric.TEXT_OUTCOMES, "success", amount=MIN_SHADOW_TEXT_REQUESTS)

    assert evaluate_shadow_gate(metrics.snapshot()).outcome == "GO"

    metrics.increment(CounterMetric.PAID_GENERATION, "attempted")
    decision = evaluate_shadow_gate(metrics.snapshot())
    assert decision.outcome == "NO-GO"
    assert "shadow_paid_generation_attempted" in decision.blockers


def _passing_canary_metrics() -> GroundedImageMetrics:
    metrics = GroundedImageMetrics(RolloutStage.CANARY)
    metrics.increment(
        CounterMetric.PAID_GENERATION, "attempted", amount=MIN_CANARY_IMAGE_RESULTS
    )
    metrics.increment(
        CounterMetric.PAID_GENERATION, "completed", amount=MIN_CANARY_IMAGE_RESULTS
    )
    metrics.increment(CounterMetric.REQUESTS, "text", amount=MIN_CANARY_TEXT_REQUESTS)
    metrics.increment(CounterMetric.ROLLBACK, "requested")
    metrics.increment(CounterMetric.ROLLBACK, "kill_switch_off")
    metrics.increment(CounterMetric.ROLLBACK, "text_available")
    for metric, value in (
        (LatencyMetric.WARM_GROUNDING, 2_000),
        (LatencyMetric.CACHED_CONTINUATION, 80),
        (LatencyMetric.FIRST_TOOL_CALL, 4_000),
        (LatencyMetric.TEXT_TOTAL, 500),
    ):
        metrics.observe_latency(metric, value)
    return metrics


def test_canary_gate_requires_every_external_gate_and_slo() -> None:
    evidence = RolloutEvidence(
        access_canary_approved=True,
        trusted_correlation_approved=True,
        production_composition_active=True,
        provenance_approved=True,
        paid_generation_approved=True,
        rollback_drill_passed=True,
        text_baseline_p95_ms=480,
    )
    decision = evaluate_canary_gate(_passing_canary_metrics().snapshot(), evidence)
    assert decision.outcome == "GO"
    assert decision.blockers == ()
    assert set(decision.slo_checks.values()) == {"pass"}


def test_canary_fails_on_unblocked_invalid_grounding_or_incomplete_rollback() -> None:
    metrics = _passing_canary_metrics()
    metrics.increment(CounterMetric.VALIDATION_FAILURES, "unblocked_violation")
    metrics.increment(CounterMetric.ROLLBACK, "failed")
    evidence = RolloutEvidence(
        access_canary_approved=True,
        trusted_correlation_approved=True,
        production_composition_active=True,
        provenance_approved=True,
        paid_generation_approved=True,
        rollback_drill_passed=True,
        text_baseline_p95_ms=480,
    )

    decision = evaluate_canary_gate(metrics.snapshot(), evidence)
    assert decision.outcome == "NO-GO"
    assert "slo_invalid_grounding_unblocked_failed" in decision.blockers
    assert "rollback_no_failure_missing_or_failed" in decision.blockers


def test_current_evidence_is_no_go_and_keeps_paid_generation_unapproved() -> None:
    current = RolloutEvidence(
        access_canary_approved=False,
        trusted_correlation_approved=False,
        production_composition_active=False,
        provenance_approved=False,
        paid_generation_approved=False,
        rollback_drill_passed=False,
    )
    decision = evaluate_canary_gate(GroundedImageMetrics().snapshot(), current)

    assert decision.outcome == "NO-GO"
    assert "access_canary_not_approved" in decision.blockers
    assert "trusted_correlation_not_approved" in decision.blockers
    assert "production_composition_inactive" in decision.blockers
    assert "paid_generation_not_approved" in decision.blockers
    assert "rollback_drill_not_passed" in decision.blockers


def test_multi_replica_requires_independent_redis_approval() -> None:
    evidence = RolloutEvidence(
        access_canary_approved=True,
        trusted_correlation_approved=True,
        production_composition_active=True,
        provenance_approved=True,
        paid_generation_approved=True,
        rollback_drill_passed=True,
        single_replica=False,
        redis_multi_replica_approved=False,
        text_baseline_p95_ms=480,
    )
    decision = evaluate_canary_gate(_passing_canary_metrics().snapshot(), evidence)
    assert decision.outcome == "NO-GO"
    assert "redis_multi_replica_not_approved" in decision.blockers


def test_dashboard_and_rollback_runbook_are_explicit_and_stay_no_go(project_root) -> None:
    readme = (project_root / "README.md").read_text(encoding="utf-8")

    assert "#### Observabilidad, shadow y SLOs del canary" in readme
    assert "100 intents de imagen" in readme
    assert "1000 requests de texto" in readme
    assert "cero llamadas al proveedor o motor de imágenes" in readme
    assert "p95 de grounding warm <= 3000 ms" in readme
    assert "p95 de continuación cacheada <= 100 ms" in readme
    assert "p95 del primer tool call <= 5000 ms" in readme
    assert "NO-GO operativo" in readme
    assert "GROUNDED_IMAGE_ENABLED=false" in readme
    assert "No ejecutar el smoke pago de 12.3" in readme
