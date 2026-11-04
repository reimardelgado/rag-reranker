from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import threading
import time

import pytest

from backend.correlation import (
    CorrelationExecutionStore,
    CorrelationLimits,
    CorrelationLoopLimitExceeded,
    CorrelationRetrievalFailed,
    CorrelationStateExpired,
    CorrelationStateMismatch,
    CorrelationStateOversized,
    DuplicateGenerationSuppressed,
    TrustedHeaderCorrelator,
)


HMAC_KEY = b"h" * 32
SERVICE_TOKEN = "test-service-token"
RETRIEVAL_FINGERPRINT = "a" * 64
PROMPT_HASH = "b" * 64


def trusted_headers(
    *, user: str = "user-a", chat: str = "chat-a", root: str = "root-a"
) -> dict[str, str]:
    return {
        "X-OpenWebUI-User-Id": user,
        "X-OpenWebUI-Chat-Id": chat,
        "X-OpenWebUI-Message-Id": root,
        "X-RAG-Service-Token": SERVICE_TOKEN,
    }


def make_store(*, clock=lambda: 0.0, max_packet_bytes: int = 256 * 1024):
    limits = CorrelationLimits(
        ttl_seconds=600,
        max_entries=64,
        max_packet_bytes=max_packet_bytes,
    )
    correlator = TrustedHeaderCorrelator(
        hmac_key=HMAC_KEY,
        service_token=SERVICE_TOKEN,
    )
    return CorrelationExecutionStore(correlator=correlator, limits=limits, clock=clock)


def initialize(store, headers=None, retrieve=lambda: {"references": ["ref-1"]}):
    return store.get_or_retrieve(
        headers or trusted_headers(),
        retrieval_fingerprint=RETRIEVAL_FINGERPRINT,
        retrieve=retrieve,
    )


def test_same_loop_reuses_detached_retrieval_without_repeating_work() -> None:
    store = make_store()
    calls = 0

    def retrieve():
        nonlocal calls
        calls += 1
        return {"references": ["ref-1"]}

    first = initialize(store, retrieve=retrieve)
    first.packet["references"].append("caller-mutation")
    second = initialize(store, retrieve=retrieve)

    assert calls == 1
    assert first.reused is False
    assert second.reused is True
    assert second.packet == {"references": ["ref-1"]}


def test_retrieval_singleflight_is_once_under_concurrent_duplicate_requests() -> None:
    store = make_store()
    calls = 0
    lock = threading.Lock()
    release = threading.Event()

    def retrieve():
        nonlocal calls
        with lock:
            calls += 1
        assert release.wait(timeout=2)
        return {"references": ["ref-1", "ref-2"]}

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(initialize, store, None, retrieve) for _ in range(8)]
        deadline = time.monotonic() + 2
        while calls == 0 and time.monotonic() < deadline:
            time.sleep(0.005)
        release.set()
        results = [future.result(timeout=2) for future in futures]

    assert calls == 1
    assert sum(not result.reused for result in results) == 1
    assert all(result.packet == {"references": ["ref-1", "ref-2"]} for result in results)


def test_retrieval_failure_is_fail_closed_and_not_retried_within_ttl() -> None:
    store = make_store()
    calls = 0

    def retrieve():
        nonlocal calls
        calls += 1
        raise TimeoutError("private upstream detail")

    with pytest.raises(TimeoutError):
        initialize(store, retrieve=retrieve)
    with pytest.raises(CorrelationRetrievalFailed, match="unavailable") as failure:
        initialize(store, retrieve=retrieve)

    assert calls == 1
    assert "private upstream detail" not in str(failure.value)


def test_retrieval_fingerprint_and_correlation_boundaries_do_not_reuse_state() -> None:
    store = make_store()
    initialize(store)

    with pytest.raises(CorrelationStateMismatch):
        store.get_or_retrieve(
            trusted_headers(),
            retrieval_fingerprint="c" * 64,
            retrieve=lambda: {"must": "not run"},
        )

    other = store.get_or_retrieve(
        trusted_headers(root="root-b"),
        retrieval_fingerprint=RETRIEVAL_FINGERPRINT,
        retrieve=lambda: {"references": ["other-loop"]},
    )
    assert other.reused is False
    assert other.packet == {"references": ["other-loop"]}


def test_normative_three_tool_iterations_reject_the_fourth_atomically() -> None:
    store = make_store()
    initialize(store)

    assert [
        store.advance_tool_iteration(
            trusted_headers(), retrieval_fingerprint=RETRIEVAL_FINGERPRINT
        )
        for _ in range(3)
    ] == [1, 2, 3]
    with pytest.raises(CorrelationLoopLimitExceeded, match="three"):
        store.advance_tool_iteration(
            trusted_headers(), retrieval_fingerprint=RETRIEVAL_FINGERPRINT
        )


def test_concurrent_iteration_race_allows_exactly_three() -> None:
    store = make_store()
    initialize(store)

    def advance():
        try:
            return store.advance_tool_iteration(
                trusted_headers(), retrieval_fingerprint=RETRIEVAL_FINGERPRINT
            )
        except CorrelationLoopLimitExceeded:
            return "blocked"

    with ThreadPoolExecutor(max_workers=12) as executor:
        outcomes = list(executor.map(lambda _: advance(), range(12)))

    assert sorted(value for value in outcomes if isinstance(value, int)) == [1, 2, 3]
    assert outcomes.count("blocked") == 9


def test_concurrent_duplicate_generation_reserves_exactly_one_paid_call() -> None:
    store = make_store()
    initialize(store)
    paid_calls = 0
    paid_lock = threading.Lock()

    def claim():
        nonlocal paid_calls
        try:
            store.claim_generation(
                trusted_headers(),
                retrieval_fingerprint=RETRIEVAL_FINGERPRINT,
                prompt_version="grounding-v1",
                prompt_hash=PROMPT_HASH,
            )
            with paid_lock:
                paid_calls += 1
            return "reserved"
        except DuplicateGenerationSuppressed:
            return "duplicate"

    with ThreadPoolExecutor(max_workers=16) as executor:
        outcomes = list(executor.map(lambda _: claim(), range(16)))

    assert outcomes.count("reserved") == 1
    assert outcomes.count("duplicate") == 15
    assert paid_calls == 1


def test_idempotency_scope_uses_user_chat_version_and_prompt_not_root_message() -> None:
    store = make_store()
    first = trusted_headers(root="root-a")
    second = trusted_headers(root="root-b")
    initialize(store, first)
    initialize(store, second)
    store.claim_generation(
        first,
        retrieval_fingerprint=RETRIEVAL_FINGERPRINT,
        prompt_version="grounding-v1",
        prompt_hash=PROMPT_HASH,
    )

    with pytest.raises(DuplicateGenerationSuppressed):
        store.claim_generation(
            first,
            retrieval_fingerprint=RETRIEVAL_FINGERPRINT,
            prompt_version="grounding-v1",
            prompt_hash=PROMPT_HASH,
        )

    # A different root has isolated retrieval/loop state, but paid-call identity
    # deliberately spans roots for the same trusted user/chat and grounded prompt.
    with pytest.raises(DuplicateGenerationSuppressed):
        store.claim_generation(
            second,
            retrieval_fingerprint=RETRIEVAL_FINGERPRINT,
            prompt_version="grounding-v1",
            prompt_hash=PROMPT_HASH,
        )


def test_expiry_allows_new_retrieval_and_generation_reservation() -> None:
    now = [0.0]
    store = make_store(clock=lambda: now[0])
    initialize(store)
    store.claim_generation(
        trusted_headers(),
        retrieval_fingerprint=RETRIEVAL_FINGERPRINT,
        prompt_version="grounding-v1",
        prompt_hash=PROMPT_HASH,
    )
    now[0] = 600.0

    with pytest.raises(CorrelationStateExpired):
        store.advance_tool_iteration(
            trusted_headers(), retrieval_fingerprint=RETRIEVAL_FINGERPRINT
        )
    refreshed = initialize(store, retrieve=lambda: {"references": ["fresh"]})
    store.claim_generation(
        trusted_headers(),
        retrieval_fingerprint=RETRIEVAL_FINGERPRINT,
        prompt_version="grounding-v1",
        prompt_hash=PROMPT_HASH,
    )
    assert refreshed.packet == {"references": ["fresh"]}


def test_execution_packet_bound_rejects_oversized_retrieval() -> None:
    store = make_store(max_packet_bytes=512)

    with pytest.raises(CorrelationStateOversized):
        initialize(store, retrieve=lambda: {"data": "x" * 512})
