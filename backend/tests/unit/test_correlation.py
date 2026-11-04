from __future__ import annotations

import pytest

from backend.correlation import (
    CorrelationLimits,
    CorrelationStateExpired,
    CorrelationStateMissing,
    CorrelationStateOversized,
    CorrelationStateStore,
    TrustedHeaderCorrelator,
    UntrustedCorrelationError,
)


HMAC_KEY = b"h" * 32
SERVICE_TOKEN = "test-service-token"


def trusted_headers(
    *, user: str = "user-a", chat: str = "chat-a", root: str = "root-a"
) -> dict[str, str]:
    return {
        "X-OpenWebUI-User-Id": user,
        "X-OpenWebUI-Chat-Id": chat,
        "X-OpenWebUI-Message-Id": root,
        "X-RAG-Service-Token": SERVICE_TOKEN,
    }


def make_store(
    *,
    clock=lambda: 0.0,
    max_entries: int = 3,
    max_packet_bytes: int = 256 * 1024,
) -> CorrelationStateStore:
    limits = CorrelationLimits(
        ttl_seconds=600,
        max_entries=max_entries,
        max_packet_bytes=max_packet_bytes,
    )
    correlator = TrustedHeaderCorrelator(
        hmac_key=HMAC_KEY,
        service_token=SERVICE_TOKEN,
        max_identifier_bytes=limits.max_identifier_bytes,
    )
    return CorrelationStateStore(correlator=correlator, limits=limits, clock=clock)


def test_hmac_key_is_stable_opaque_and_domain_separates_tuple_parts() -> None:
    correlator = TrustedHeaderCorrelator(
        hmac_key=HMAC_KEY,
        service_token=SERVICE_TOKEN,
    )
    headers = trusted_headers()

    first = correlator.derive_key(headers)
    second = correlator.derive_key({key.lower(): value for key, value in headers.items()})
    ambiguous_without_framing = correlator.derive_key(
        trusted_headers(user="user-ach", chat="at-a", root="root-a")
    )

    assert first == second
    assert first != ambiguous_without_framing
    assert len(first.digest) == 64
    assert all(value not in repr(first) for value in ("user-a", "chat-a", "root-a"))
    assert HMAC_KEY.decode() not in repr(correlator)
    assert SERVICE_TOKEN not in repr(correlator)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda headers: headers.pop("X-RAG-Service-Token"),
        lambda headers: headers.__setitem__("X-RAG-Service-Token", "tampered"),
        lambda headers: headers.pop("X-OpenWebUI-User-Id"),
        lambda headers: headers.__setitem__("X-OpenWebUI-Chat-Id", " bad "),
        lambda headers: headers.__setitem__("X-OpenWebUI-Message-Id", "bad\r\nvalue"),
    ],
)
def test_missing_tampered_or_malformed_trusted_headers_fail_closed(mutate) -> None:
    correlator = TrustedHeaderCorrelator(
        hmac_key=HMAC_KEY,
        service_token=SERVICE_TOKEN,
    )
    headers = trusted_headers()
    mutate(headers)

    with pytest.raises(UntrustedCorrelationError):
        correlator.derive_key(headers)


def test_exact_user_chat_root_tuple_is_required_and_repeated_reads_are_detached() -> None:
    store = make_store()
    original = {"prompt_hash": "0" * 64, "references": ["ref-1"]}
    store.put(trusted_headers(), original)
    original["references"].append("caller-mutation")

    first = store.get(trusted_headers())
    first["references"].append("reader-mutation")
    second = store.get(trusted_headers())

    assert second == {"prompt_hash": "0" * 64, "references": ["ref-1"]}
    for changed in (
        trusted_headers(user="user-b"),
        trusted_headers(chat="chat-b"),
        trusted_headers(root="root-b"),
    ):
        with pytest.raises(CorrelationStateMissing):
            store.get(changed)


def test_state_expires_at_normative_ten_minute_boundary() -> None:
    now = [100.0]
    store = make_store(clock=lambda: now[0])
    store.put(trusted_headers(), {"state": "bounded"})

    now[0] += 599.999
    assert store.get(trusted_headers()) == {"state": "bounded"}
    now[0] += 0.001
    with pytest.raises(CorrelationStateExpired):
        store.get(trusted_headers())
    with pytest.raises(CorrelationStateMissing):
        store.get(trusted_headers())


def test_lru_access_protects_recent_entry_and_evicts_oldest() -> None:
    store = make_store(max_entries=2)
    first = trusted_headers(root="root-1")
    second = trusted_headers(root="root-2")
    third = trusted_headers(root="root-3")
    store.put(first, {"slot": 1})
    store.put(second, {"slot": 2})
    assert store.get(first) == {"slot": 1}

    store.put(third, {"slot": 3})

    with pytest.raises(CorrelationStateMissing):
        store.get(second)
    assert store.get(first) == {"slot": 1}
    assert store.get(third) == {"slot": 3}
    assert len(store) == 2


def test_packet_limit_is_exactly_256_kib_and_rejection_does_not_replace_state() -> None:
    store = make_store()
    headers = trusted_headers()
    exact_payload = "x" * (256 * 1024 - len('{"data":""}'))
    store.put(headers, {"data": exact_payload})
    assert len(store.get(headers)["data"]) == len(exact_payload)

    with pytest.raises(CorrelationStateOversized):
        store.put(headers, {"data": exact_payload + "x"})
    assert store.get(headers) == {"data": exact_payload}


def test_non_json_and_non_finite_state_fail_closed_without_cache_entry() -> None:
    store = make_store()
    for state in ({"bad": object()}, {"bad": float("nan")}):
        with pytest.raises(CorrelationStateOversized):
            store.put(trusted_headers(), state)
    assert len(store) == 0


def test_expired_entries_do_not_consume_lru_capacity() -> None:
    now = [0.0]
    store = make_store(clock=lambda: now[0], max_entries=1)
    store.put(trusted_headers(root="old"), {"slot": "old"})
    now[0] = 600.0
    store.put(trusted_headers(root="new"), {"slot": "new"})

    assert len(store) == 1
    assert store.get(trusted_headers(root="new")) == {"slot": "new"}
    with pytest.raises(CorrelationStateMissing):
        store.get(trusted_headers(root="old"))
