"""Bounded, redacted observability contracts for grounded-image rollout.

The module is deliberately inactive: ``backend.app`` does not import or compose
it.  It stores only fixed-cardinality aggregate counters and histogram buckets;
request identifiers, trusted identifiers, prompts, evidence, credentials, and
provider-private metadata are not accepted by the API.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import threading
from types import MappingProxyType
from typing import Mapping


class RolloutStage(str, Enum):
    INACTIVE = "inactive"
    SHADOW = "shadow"
    CANARY = "canary"


class CounterMetric(str, Enum):
    REQUESTS = "grounded_image_requests_total"
    DECISIONS = "grounded_image_decisions_total"
    VALIDATION_FAILURES = "grounded_image_validation_failures_total"
    DUPLICATES = "grounded_image_duplicates_total"
    CORRELATION = "grounded_image_correlation_total"
    PROVIDER_OUTCOMES = "grounded_image_provider_outcomes_total"
    PAID_GENERATION = "grounded_image_paid_generation_total"
    RETRIEVAL_REUSE = "grounded_image_retrieval_reuse_total"
    TEXT_OUTCOMES = "grounded_image_text_outcomes_total"
    ROLLBACK = "grounded_image_rollback_total"
    SHADOW = "grounded_image_shadow_total"


class LatencyMetric(str, Enum):
    TEXT_TOTAL = "grounded_image_text_total_latency_ms"
    RETRIEVAL = "grounded_image_retrieval_latency_ms"
    RERANK = "grounded_image_rerank_latency_ms"
    WARM_GROUNDING = "grounded_image_warm_grounding_latency_ms"
    CACHED_CONTINUATION = "grounded_image_cached_continuation_latency_ms"
    PROVIDER_FIRST_BYTE = "grounded_image_provider_first_byte_latency_ms"
    FIRST_TOOL_CALL = "grounded_image_first_tool_call_latency_ms"
    IMAGE_EXECUTION = "grounded_image_execution_latency_ms"


_COUNTER_OUTCOMES: Mapping[CounterMetric, frozenset[str]] = MappingProxyType(
    {
        CounterMetric.REQUESTS: frozenset({"text", "image_intent"}),
        CounterMetric.DECISIONS: frozenset(
            {"eligible", "shadow_eligible", "refused", "denied", "invalid"}
        ),
        CounterMetric.VALIDATION_FAILURES: frozenset(
            {
                "blocked_grounding",
                "blocked_tool_call",
                "blocked_provenance",
                "unblocked_violation",
            }
        ),
        CounterMetric.DUPLICATES: frozenset(
            {"suppressed", "paid_call_violation"}
        ),
        CounterMetric.CORRELATION: frozenset(
            {"resolved", "expired", "mismatch", "oversized", "cross_user_violation"}
        ),
        CounterMetric.PROVIDER_OUTCOMES: frozenset(
            {"success", "timeout", "rate_limited", "unavailable", "incompatible"}
        ),
        CounterMetric.PAID_GENERATION: frozenset({"attempted", "completed", "failed"}),
        CounterMetric.RETRIEVAL_REUSE: frozenset(
            {"reused", "fresh", "same_loop_repeat_violation"}
        ),
        CounterMetric.TEXT_OUTCOMES: frozenset({"success", "failure", "parity_failure"}),
        CounterMetric.ROLLBACK: frozenset(
            {"requested", "kill_switch_off", "text_available", "failed"}
        ),
        CounterMetric.SHADOW: frozenset(
            {"evaluated", "provider_or_image_attempt_violation"}
        ),
    }
)

LATENCY_BUCKETS_MS = (
    10,
    25,
    50,
    100,
    200,
    500,
    1_000,
    3_000,
    5_000,
    10_000,
    30_000,
    60_000,
)
MAX_LATENCY_MS = LATENCY_BUCKETS_MS[-1]
MAX_COUNTER_INCREMENT = 1_000_000

MIN_SHADOW_IMAGE_INTENTS = 100
MIN_SHADOW_TEXT_REQUESTS = 1_000
MIN_CANARY_IMAGE_RESULTS = 20
MIN_CANARY_TEXT_REQUESTS = 500


class MetricsContractError(ValueError):
    """A safe observability-contract failure that never echoes supplied data."""


@dataclass(slots=True)
class _Histogram:
    buckets: list[int]
    count: int = 0
    total_ms: float = 0.0
    maximum_ms: float = 0.0

    def observe(self, value_ms: float) -> None:
        self.count += 1
        self.total_ms += value_ms
        self.maximum_ms = max(self.maximum_ms, value_ms)
        for index, upper_bound in enumerate(LATENCY_BUCKETS_MS):
            if value_ms <= upper_bound:
                self.buckets[index] += 1
                return

    def snapshot(self) -> dict[str, object]:
        percentile = None
        if self.count:
            rank = math.ceil(self.count * 0.95)
            cumulative = 0
            for upper_bound, bucket_count in zip(LATENCY_BUCKETS_MS, self.buckets):
                cumulative += bucket_count
                if cumulative >= rank:
                    percentile = upper_bound
                    break
        return {
            "count": self.count,
            "sum_ms": round(self.total_ms, 3),
            "max_ms": round(self.maximum_ms, 3),
            "p95_upper_bound_ms": percentile,
            "buckets": {
                str(bound): count
                for bound, count in zip(LATENCY_BUCKETS_MS, self.buckets)
            },
        }


class GroundedImageMetrics:
    """Thread-safe, process-local fixed-cardinality aggregate recorder."""

    def __init__(self, stage: RolloutStage = RolloutStage.INACTIVE) -> None:
        if not isinstance(stage, RolloutStage):
            raise MetricsContractError("Rollout stage is invalid.")
        self._stage = stage
        self._counters = {
            (metric, outcome): 0
            for metric, outcomes in _COUNTER_OUTCOMES.items()
            for outcome in outcomes
        }
        self._latencies = {
            metric: _Histogram([0] * len(LATENCY_BUCKETS_MS))
            for metric in LatencyMetric
        }
        self._lock = threading.Lock()

    @property
    def stage(self) -> RolloutStage:
        return self._stage

    def increment(
        self, metric: CounterMetric, outcome: str, *, amount: int = 1
    ) -> None:
        if (
            not isinstance(metric, CounterMetric)
            or not isinstance(outcome, str)
            or outcome not in _COUNTER_OUTCOMES.get(metric, frozenset())
            or not isinstance(amount, int)
            or isinstance(amount, bool)
            or not 1 <= amount <= MAX_COUNTER_INCREMENT
        ):
            raise MetricsContractError("Counter observation is outside the contract.")
        with self._lock:
            self._counters[(metric, outcome)] += amount

    def observe_latency(self, metric: LatencyMetric, value_ms: float) -> None:
        if (
            not isinstance(metric, LatencyMetric)
            or isinstance(value_ms, bool)
            or not isinstance(value_ms, (int, float))
            or not math.isfinite(value_ms)
            or not 0 <= value_ms <= MAX_LATENCY_MS
        ):
            raise MetricsContractError("Latency observation is outside the contract.")
        with self._lock:
            self._latencies[metric].observe(float(value_ms))

    def snapshot(self) -> dict[str, object]:
        """Return detached dashboard data with no per-request dimensions."""

        with self._lock:
            counters = {
                metric.value: {
                    outcome: self._counters[(metric, outcome)]
                    for outcome in sorted(outcomes)
                }
                for metric, outcomes in _COUNTER_OUTCOMES.items()
            }
            latencies = {
                metric.value: histogram.snapshot()
                for metric, histogram in self._latencies.items()
            }
        return {
            "schema_version": 1,
            "stage": self._stage.value,
            "counters": counters,
            "latencies": latencies,
            "redaction": "fixed-cardinality-aggregates-only",
        }


@dataclass(frozen=True, slots=True)
class RolloutEvidence:
    """Redacted release evidence; booleans never carry identities or credentials."""

    access_canary_approved: bool = False
    trusted_correlation_approved: bool = False
    production_composition_active: bool = False
    provenance_approved: bool = False
    native_adapter_approved: bool = True
    single_replica: bool = True
    redis_multi_replica_approved: bool = False
    paid_generation_approved: bool = False
    rollback_drill_passed: bool = False
    text_baseline_p95_ms: float | None = None


@dataclass(frozen=True, slots=True)
class RolloutDecision:
    outcome: str
    blockers: tuple[str, ...]
    slo_checks: Mapping[str, str]


def _counter(snapshot: Mapping[str, object], metric: CounterMetric, outcome: str) -> int:
    try:
        value = snapshot["counters"][metric.value][outcome]  # type: ignore[index]
    except (KeyError, TypeError):
        raise MetricsContractError("Metrics snapshot does not match the contract.") from None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise MetricsContractError("Metrics snapshot does not match the contract.")
    return value


def _p95(snapshot: Mapping[str, object], metric: LatencyMetric) -> int | None:
    try:
        value = snapshot["latencies"][metric.value]["p95_upper_bound_ms"]  # type: ignore[index]
    except (KeyError, TypeError):
        raise MetricsContractError("Metrics snapshot does not match the contract.") from None
    if value is not None and (not isinstance(value, int) or isinstance(value, bool)):
        raise MetricsContractError("Metrics snapshot does not match the contract.")
    return value


def evaluate_shadow_gate(snapshot: Mapping[str, object]) -> RolloutDecision:
    """Evaluate a no-provider/no-image shadow window conservatively."""

    blockers: list[str] = []
    if snapshot.get("stage") != RolloutStage.SHADOW.value:
        blockers.append("shadow_stage_not_proven")
    if _counter(snapshot, CounterMetric.SHADOW, "evaluated") < MIN_SHADOW_IMAGE_INTENTS:
        blockers.append("shadow_image_sample_insufficient")
    if _counter(snapshot, CounterMetric.REQUESTS, "text") < MIN_SHADOW_TEXT_REQUESTS:
        blockers.append("shadow_text_sample_insufficient")
    if _counter(snapshot, CounterMetric.PAID_GENERATION, "attempted") != 0:
        blockers.append("shadow_paid_generation_attempted")
    if _counter(snapshot, CounterMetric.SHADOW, "provider_or_image_attempt_violation") != 0:
        blockers.append("shadow_provider_or_image_attempted")
    if _counter(snapshot, CounterMetric.TEXT_OUTCOMES, "parity_failure") != 0:
        blockers.append("shadow_text_parity_failed")
    if _counter(snapshot, CounterMetric.CORRELATION, "cross_user_violation") != 0:
        blockers.append("cross_user_isolation_failed")
    return RolloutDecision(
        outcome="GO" if not blockers else "NO-GO",
        blockers=tuple(blockers),
        slo_checks=MappingProxyType({"shadow_no_paid_or_provider_calls": "pass" if not blockers else "fail"}),
    )


def evaluate_canary_gate(
    snapshot: Mapping[str, object], evidence: RolloutEvidence
) -> RolloutDecision:
    """Evaluate release gates without activating or authorizing the paid path."""

    blockers: list[str] = []
    required_evidence = {
        "access_canary_not_approved": evidence.access_canary_approved,
        "trusted_correlation_not_approved": evidence.trusted_correlation_approved,
        "production_composition_inactive": evidence.production_composition_active,
        "provenance_not_approved": evidence.provenance_approved,
        "native_adapter_not_approved": evidence.native_adapter_approved,
        "paid_generation_not_approved": evidence.paid_generation_approved,
        "rollback_drill_not_passed": evidence.rollback_drill_passed,
    }
    blockers.extend(name for name, passed in required_evidence.items() if passed is not True)
    if not evidence.single_replica and not evidence.redis_multi_replica_approved:
        blockers.append("redis_multi_replica_not_approved")
    if snapshot.get("stage") != RolloutStage.CANARY.value:
        blockers.append("canary_stage_not_proven")
    paid_attempted = _counter(snapshot, CounterMetric.PAID_GENERATION, "attempted")
    paid_completed = _counter(snapshot, CounterMetric.PAID_GENERATION, "completed")
    paid_failed = _counter(snapshot, CounterMetric.PAID_GENERATION, "failed")
    if paid_completed < MIN_CANARY_IMAGE_RESULTS:
        blockers.append("canary_image_sample_insufficient")
    if paid_completed + paid_failed > paid_attempted:
        blockers.append("paid_generation_accounting_invalid")
    if _counter(snapshot, CounterMetric.REQUESTS, "text") < MIN_CANARY_TEXT_REQUESTS:
        blockers.append("canary_text_sample_insufficient")

    slo_checks: dict[str, str] = {}
    zero_metrics = {
        "cross_user_leaks": (CounterMetric.CORRELATION, "cross_user_violation"),
        "duplicate_paid_calls": (CounterMetric.DUPLICATES, "paid_call_violation"),
        "invalid_grounding_unblocked": (
            CounterMetric.VALIDATION_FAILURES,
            "unblocked_violation",
        ),
        "same_loop_retrieval_repeats": (
            CounterMetric.RETRIEVAL_REUSE,
            "same_loop_repeat_violation",
        ),
    }
    for name, (metric, outcome) in zero_metrics.items():
        passed = _counter(snapshot, metric, outcome) == 0
        slo_checks[name] = "pass" if passed else "fail"
        if not passed:
            blockers.append(f"slo_{name}_failed")

    rollback_checks = {
        "rollback_requested": _counter(snapshot, CounterMetric.ROLLBACK, "requested") > 0,
        "kill_switch_off": _counter(snapshot, CounterMetric.ROLLBACK, "kill_switch_off") > 0,
        "rollback_text_available": _counter(
            snapshot, CounterMetric.ROLLBACK, "text_available"
        ) > 0,
        "rollback_no_failure": _counter(snapshot, CounterMetric.ROLLBACK, "failed") == 0,
    }
    for name, passed in rollback_checks.items():
        slo_checks[name] = "pass" if passed else "missing_or_fail"
        if not passed:
            blockers.append(f"{name}_missing_or_failed")

    latency_limits = {
        "warm_grounding_p95": (LatencyMetric.WARM_GROUNDING, 3_000),
        "cached_continuation_p95": (LatencyMetric.CACHED_CONTINUATION, 100),
        "first_tool_call_p95": (LatencyMetric.FIRST_TOOL_CALL, 5_000),
    }
    for name, (metric, limit) in latency_limits.items():
        value = _p95(snapshot, metric)
        passed = value is not None and value <= limit
        slo_checks[name] = "pass" if passed else "missing_or_fail"
        if not passed:
            blockers.append(f"slo_{name}_missing_or_failed")

    text_p95 = _p95(snapshot, LatencyMetric.TEXT_TOTAL)
    baseline = evidence.text_baseline_p95_ms
    text_passed = (
        text_p95 is not None
        and baseline is not None
        and math.isfinite(baseline)
        and baseline > 0
        and (
            text_p95 - baseline <= 200
            or (text_p95 - baseline) / baseline <= 0.10
        )
    )
    slo_checks["text_p95_regression"] = "pass" if text_passed else "missing_or_fail"
    if not text_passed:
        blockers.append("slo_text_p95_regression_missing_or_failed")

    return RolloutDecision(
        outcome="GO" if not blockers else "NO-GO",
        blockers=tuple(dict.fromkeys(blockers)),
        slo_checks=MappingProxyType(slo_checks),
    )
