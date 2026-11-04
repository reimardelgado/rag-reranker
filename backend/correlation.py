"""Bounded, process-local correlation state for trusted continuation headers.

This module is deliberately inactive: production composition does not import it.
Raw trusted identifiers are used only while deriving a domain-separated HMAC key;
the cache retains neither identifiers nor authentication values and performs no
logging or durable persistence.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import hmac
import json
import math
import secrets
import threading
import time
from typing import Any


class CorrelationError(ValueError):
    """Base class for bounded, safe correlation failures."""


class UntrustedCorrelationError(CorrelationError):
    """Trusted headers or their service authentication are invalid."""


class CorrelationStateMissing(CorrelationError):
    """No state exists for this exact trusted correlation tuple."""


class CorrelationStateExpired(CorrelationError):
    """State existed but exceeded its configured TTL."""


class CorrelationStateOversized(CorrelationError):
    """State cannot fit within the configured per-packet bound."""


class CorrelationStateMismatch(CorrelationError):
    """State belongs to a different retrieval within the correlation loop."""


class CorrelationRetrievalFailed(CorrelationError):
    """The loop's one retrieval attempt failed and cannot be replayed."""


class DuplicateGenerationSuppressed(CorrelationError):
    """This grounded prompt already reserved its one paid call for the TTL."""


class CorrelationLoopLimitExceeded(CorrelationError):
    """The normative three tool iterations have already been consumed."""


MAX_TOOL_ITERATIONS = 3


@dataclass(frozen=True, slots=True)
class CorrelationLimits:
    """Normative process-local state limits for one replica."""

    ttl_seconds: float = 10 * 60
    max_entries: int = 1024
    max_packet_bytes: int = 256 * 1024
    max_identifier_bytes: int = 1024

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.ttl_seconds)
            or self.ttl_seconds <= 0
            or self.max_entries <= 0
            or self.max_packet_bytes <= 0
            or self.max_identifier_bytes <= 0
        ):
            raise ValueError("Correlation bounds must be positive and finite.")


@dataclass(frozen=True, slots=True)
class CorrelationKey:
    """Opaque HMAC digest; it never contains a raw trusted identifier."""

    digest: str

    def __post_init__(self) -> None:
        if len(self.digest) != 64:
            raise ValueError("Correlation digest has an invalid shape.")
        try:
            bytes.fromhex(self.digest)
        except ValueError:
            raise ValueError("Correlation digest has an invalid shape.") from None


@dataclass(frozen=True, slots=True)
class _CacheEntry:
    packet: bytes
    expires_at: float


@dataclass(slots=True)
class _ExecutionEntry:
    retrieval_fingerprint: str
    expires_at: float
    retrieval_packet: bytes | None = None
    retrieval_inflight: bool = True
    retrieval_failed: bool = False
    tool_iterations: int = 0


@dataclass(frozen=True, slots=True)
class RetrievalReuse:
    """Detached retrieval data plus whether this call reused loop state."""

    packet: Any
    reused: bool


class TrustedHeaderCorrelator:
    """Authenticate trusted headers and derive an opaque correlation HMAC.

    The service token is an independent trust gate.  Possession of user/chat
    identifiers alone is never enough to resolve state.
    """

    def __init__(
        self,
        *,
        hmac_key: bytes,
        service_token: str,
        user_header: str = "x-openwebui-user-id",
        chat_header: str = "x-openwebui-chat-id",
        root_message_header: str = "x-openwebui-message-id",
        service_token_header: str = "x-rag-service-token",
        max_identifier_bytes: int = 1024,
    ) -> None:
        if not isinstance(hmac_key, bytes) or len(hmac_key) < 32:
            raise ValueError("Correlation HMAC key must contain at least 32 bytes.")
        if not isinstance(service_token, str) or not service_token:
            raise ValueError("Trusted service token is required.")
        if max_identifier_bytes <= 0:
            raise ValueError("Identifier bound must be positive.")
        header_names = (
            user_header,
            chat_header,
            root_message_header,
            service_token_header,
        )
        if any(not isinstance(name, str) or not name.strip() for name in header_names):
            raise ValueError("Trusted header names must be non-empty strings.")
        normalized_names = tuple(name.casefold() for name in header_names)
        if len(set(normalized_names)) != len(normalized_names):
            raise ValueError("Trusted header names must be distinct.")

        self._hmac_key = hmac_key
        self._service_token = service_token
        self._identity_headers = normalized_names[:3]
        self._service_token_header = normalized_names[3]
        self._max_identifier_bytes = max_identifier_bytes

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(identity_headers={self._identity_headers!r}, "
            f"service_token_header={self._service_token_header!r}, "
            f"max_identifier_bytes={self._max_identifier_bytes!r})"
        )

    def derive_key(self, headers: Mapping[str, str]) -> CorrelationKey:
        normalized = self._normalize_headers(headers)
        values = self._authenticated_identity(normalized)
        framed = bytearray(b"rag-correlation-v1\x00")
        for value in values:
            encoded = value.encode("utf-8")
            framed.extend(len(encoded).to_bytes(4, "big"))
            framed.extend(encoded)
        digest = hmac.new(self._hmac_key, framed, hashlib.sha256).hexdigest()
        return CorrelationKey(digest=digest)

    def derive_idempotency_key(
        self,
        headers: Mapping[str, str],
        *,
        prompt_version: str,
        prompt_hash: str,
    ) -> str:
        """Derive the designed user/chat/version/hash key without retaining inputs."""

        normalized = self._normalize_headers(headers)
        values = self._authenticated_identity(normalized)
        version = self._validated_component(prompt_version, max_bytes=128)
        validated_hash = self._validated_sha256(prompt_hash, "Prompt hash")

        identity_frame = bytearray(b"rag-idempotency-identity-v1\x00")
        for value in values[:2]:
            encoded = value.encode("utf-8")
            identity_frame.extend(len(encoded).to_bytes(4, "big"))
            identity_frame.extend(encoded)
        identity_digest = hmac.new(
            self._hmac_key, identity_frame, hashlib.sha256
        ).digest()

        frame = bytearray(b"rag-generation-idempotency-v1\x00")
        for value in (identity_digest, version.encode("utf-8"), bytes.fromhex(validated_hash)):
            frame.extend(len(value).to_bytes(4, "big"))
            frame.extend(value)
        return hashlib.sha256(frame).hexdigest()

    def _authenticated_identity(self, normalized: Mapping[str, str]) -> tuple[str, ...]:
        supplied_token = normalized.get(self._service_token_header)
        if not isinstance(supplied_token, str) or not secrets.compare_digest(
            supplied_token, self._service_token
        ):
            raise UntrustedCorrelationError("Trusted correlation authentication failed.")
        return tuple(
            self._validated_identifier(normalized.get(name))
            for name in self._identity_headers
        )

    @staticmethod
    def _validated_component(value: object, *, max_bytes: int) -> str:
        if not isinstance(value, str) or not value or value != value.strip():
            raise CorrelationStateMismatch("Idempotency input is invalid.")
        if any(ord(character) < 0x20 or ord(character) == 0x7F for character in value):
            raise CorrelationStateMismatch("Idempotency input is invalid.")
        if len(value.encode("utf-8")) > max_bytes:
            raise CorrelationStateMismatch("Idempotency input exceeds its bound.")
        return value

    @staticmethod
    def _validated_sha256(value: object, label: str) -> str:
        if not isinstance(value, str) or len(value) != 64:
            raise CorrelationStateMismatch(f"{label} has an invalid shape.")
        try:
            bytes.fromhex(value)
        except ValueError:
            raise CorrelationStateMismatch(f"{label} has an invalid shape.") from None
        return value.lower()

    def _normalize_headers(self, headers: Mapping[str, str]) -> dict[str, str]:
        if not isinstance(headers, Mapping):
            raise UntrustedCorrelationError("Trusted correlation headers are required.")
        normalized: dict[str, str] = {}
        required = set(self._identity_headers) | {self._service_token_header}
        for name, value in headers.items():
            if not isinstance(name, str) or not isinstance(value, str):
                raise UntrustedCorrelationError("Trusted correlation headers are malformed.")
            key = name.casefold()
            if key not in required:
                continue
            if key in normalized:
                raise UntrustedCorrelationError("Duplicate trusted headers are not allowed.")
            normalized[key] = value
        return normalized

    def _validated_identifier(self, value: object) -> str:
        if not isinstance(value, str) or not value or value != value.strip():
            raise UntrustedCorrelationError("Trusted correlation identity is invalid.")
        if any(ord(character) < 0x20 or ord(character) == 0x7F for character in value):
            raise UntrustedCorrelationError("Trusted correlation identity is invalid.")
        if len(value.encode("utf-8")) > self._max_identifier_bytes:
            raise UntrustedCorrelationError("Trusted correlation identity exceeds its bound.")
        return value


class CorrelationStateStore:
    """A JSON-only TTL/LRU cache isolated by trusted-header HMAC tuples."""

    def __init__(
        self,
        *,
        correlator: TrustedHeaderCorrelator,
        limits: CorrelationLimits | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._correlator = correlator
        self._limits = limits or CorrelationLimits()
        self._clock = clock
        self._entries: OrderedDict[CorrelationKey, _CacheEntry] = OrderedDict()
        self._lock = threading.RLock()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(limits={self._limits!r}, entries={len(self)!r})"

    def __len__(self) -> int:
        with self._lock:
            self._purge_expired(self._now())
            return len(self._entries)

    def put(self, headers: Mapping[str, str], state: Any) -> None:
        """Insert a detached JSON packet, refreshing this tuple's TTL and LRU age."""

        key = self._correlator.derive_key(headers)
        packet = self._encode_packet(state)
        now = self._now()
        with self._lock:
            self._purge_expired(now)
            self._entries.pop(key, None)
            self._entries[key] = _CacheEntry(
                packet=packet,
                expires_at=now + self._limits.ttl_seconds,
            )
            while len(self._entries) > self._limits.max_entries:
                self._entries.popitem(last=False)

    def get(self, headers: Mapping[str, str]) -> Any:
        """Return a detached packet only for the exact authenticated tuple."""

        key = self._correlator.derive_key(headers)
        now = self._now()
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None and now >= entry.expires_at:
                del self._entries[key]
                self._purge_expired(now)
                raise CorrelationStateExpired("Correlation state has expired.")
            self._purge_expired(now)
            if entry is None:
                raise CorrelationStateMissing("Correlation state is unavailable.")
            self._entries.move_to_end(key)
            packet = entry.packet
        return json.loads(packet)

    def discard(self, headers: Mapping[str, str]) -> bool:
        """Remove only the exact authenticated tuple, if present and unexpired."""

        key = self._correlator.derive_key(headers)
        now = self._now()
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None and now >= entry.expires_at:
                del self._entries[key]
                self._purge_expired(now)
                raise CorrelationStateExpired("Correlation state has expired.")
            self._purge_expired(now)
            return self._entries.pop(key, None) is not None

    def _encode_packet(self, state: Any) -> bytes:
        try:
            packet = json.dumps(
                state,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        except (TypeError, ValueError):
            raise CorrelationStateOversized(
                "Correlation state must be bounded JSON data."
            ) from None
        if len(packet) > self._limits.max_packet_bytes:
            raise CorrelationStateOversized(
                "Correlation state exceeds the configured packet bound."
            )
        return packet

    def _now(self) -> float:
        value = self._clock()
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise RuntimeError("Correlation clock returned an invalid value.")
        return float(value)

    def _purge_expired(self, now: float) -> None:
        expired = [
            key for key, entry in self._entries.items() if now >= entry.expires_at
        ]
        for key in expired:
            del self._entries[key]


class CorrelationExecutionStore:
    """Atomic process-local execution guards for one correlated tool loop.

    Retrieval work runs outside the lock.  Concurrent callers for the same loop
    wait for its single attempt, while unrelated correlations can retrieve in
    parallel.  This class is intentionally not safe across processes or replicas.
    """

    def __init__(
        self,
        *,
        correlator: TrustedHeaderCorrelator,
        limits: CorrelationLimits | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._correlator = correlator
        self._limits = limits or CorrelationLimits()
        self._clock = clock
        self._entries: OrderedDict[CorrelationKey, _ExecutionEntry] = OrderedDict()
        self._generation_reservations: OrderedDict[str, float] = OrderedDict()
        self._condition = threading.Condition(threading.RLock())

    def __repr__(self) -> str:
        return f"{type(self).__name__}(limits={self._limits!r}, entries={len(self)!r})"

    def __len__(self) -> int:
        with self._condition:
            self._purge_expired(self._now())
            return len(self._entries)

    def get_or_retrieve(
        self,
        headers: Mapping[str, str],
        *,
        retrieval_fingerprint: str,
        retrieve: Callable[[], Any],
    ) -> RetrievalReuse:
        """Run retrieval at most once for an exact loop and return detached JSON."""

        key = self._correlator.derive_key(headers)
        fingerprint = TrustedHeaderCorrelator._validated_sha256(
            retrieval_fingerprint, "Retrieval fingerprint"
        )
        owner = False
        with self._condition:
            while True:
                now = self._now()
                self._purge_expired(now)
                entry = self._entries.get(key)
                if entry is None:
                    entry = _ExecutionEntry(
                        retrieval_fingerprint=fingerprint,
                        expires_at=now + self._limits.ttl_seconds,
                    )
                    self._entries[key] = entry
                    self._evict_lru()
                    owner = True
                    break
                if entry.retrieval_fingerprint != fingerprint:
                    raise CorrelationStateMismatch(
                        "Correlation loop retrieval does not match cached state."
                    )
                self._entries.move_to_end(key)
                if entry.retrieval_packet is not None:
                    return RetrievalReuse(packet=json.loads(entry.retrieval_packet), reused=True)
                if entry.retrieval_failed:
                    raise CorrelationRetrievalFailed(
                        "Correlation loop retrieval is unavailable."
                    )
                self._condition.wait()

        if not owner:  # pragma: no cover - the loop exits only for its owner
            raise RuntimeError("Retrieval ownership invariant failed.")
        try:
            packet = self._encode_packet(retrieve())
        except BaseException:
            with self._condition:
                current = self._entries.get(key)
                if current is entry:
                    entry.retrieval_inflight = False
                    entry.retrieval_failed = True
                    self._condition.notify_all()
            raise

        with self._condition:
            current = self._entries.get(key)
            if current is not entry:
                raise CorrelationStateExpired("Correlation state expired during retrieval.")
            entry.retrieval_packet = packet
            entry.retrieval_inflight = False
            try:
                self._ensure_entry_bound(entry)
            except CorrelationStateOversized:
                entry.retrieval_packet = None
                entry.retrieval_failed = True
                self._condition.notify_all()
                raise
            self._condition.notify_all()
        return RetrievalReuse(packet=json.loads(packet), reused=False)

    def advance_tool_iteration(
        self,
        headers: Mapping[str, str],
        *,
        retrieval_fingerprint: str,
    ) -> int:
        """Atomically consume one of exactly three allowed tool iterations."""

        key, entry = self._resolve_ready_entry(headers, retrieval_fingerprint)
        with self._condition:
            self._assert_current(key, entry)
            if entry.tool_iterations >= MAX_TOOL_ITERATIONS:
                raise CorrelationLoopLimitExceeded(
                    "Correlation loop exceeded three tool iterations."
                )
            entry.tool_iterations += 1
            self._ensure_entry_bound(entry)
            return entry.tool_iterations

    def claim_generation(
        self,
        headers: Mapping[str, str],
        *,
        retrieval_fingerprint: str,
        prompt_version: str,
        prompt_hash: str,
    ) -> None:
        """Atomically reserve one paid generation for this key for the full TTL."""

        idempotency_key = self._correlator.derive_idempotency_key(
            headers,
            prompt_version=prompt_version,
            prompt_hash=prompt_hash,
        )
        key, entry = self._resolve_ready_entry(headers, retrieval_fingerprint)
        with self._condition:
            self._assert_current(key, entry)
            now = self._now()
            self._purge_generation_reservations(now)
            if idempotency_key in self._generation_reservations:
                raise DuplicateGenerationSuppressed(
                    "Duplicate paid generation was suppressed."
                )
            if len(self._generation_reservations) >= self._limits.max_entries:
                raise CorrelationStateOversized(
                    "Generation reservation capacity is unavailable."
                )
            self._generation_reservations[idempotency_key] = (
                now + self._limits.ttl_seconds
            )

    def _resolve_ready_entry(
        self,
        headers: Mapping[str, str],
        retrieval_fingerprint: str,
    ) -> tuple[CorrelationKey, _ExecutionEntry]:
        key = self._correlator.derive_key(headers)
        fingerprint = TrustedHeaderCorrelator._validated_sha256(
            retrieval_fingerprint, "Retrieval fingerprint"
        )
        with self._condition:
            now = self._now()
            entry = self._entries.get(key)
            if entry is not None and now >= entry.expires_at:
                del self._entries[key]
                self._condition.notify_all()
                raise CorrelationStateExpired("Correlation state has expired.")
            self._purge_expired(now)
            if entry is None:
                raise CorrelationStateMissing("Correlation state is unavailable.")
            if entry.retrieval_fingerprint != fingerprint:
                raise CorrelationStateMismatch(
                    "Correlation loop retrieval does not match cached state."
                )
            if entry.retrieval_failed:
                raise CorrelationRetrievalFailed(
                    "Correlation loop retrieval is unavailable."
                )
            if entry.retrieval_inflight or entry.retrieval_packet is None:
                raise CorrelationStateMissing("Correlation retrieval is not ready.")
            self._entries.move_to_end(key)
            return key, entry

    def _assert_current(self, key: CorrelationKey, entry: _ExecutionEntry) -> None:
        now = self._now()
        if now >= entry.expires_at or self._entries.get(key) is not entry:
            self._entries.pop(key, None)
            raise CorrelationStateExpired("Correlation state has expired.")

    def _ensure_entry_bound(self, entry: _ExecutionEntry) -> None:
        state = {
            "retrieval": json.loads(entry.retrieval_packet)
            if entry.retrieval_packet is not None
            else None,
            "retrieval_failed": entry.retrieval_failed,
            "retrieval_fingerprint": entry.retrieval_fingerprint,
            "tool_iterations": entry.tool_iterations,
        }
        self._encode_packet(state)

    def _encode_packet(self, state: Any) -> bytes:
        try:
            packet = json.dumps(
                state,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        except (TypeError, ValueError):
            raise CorrelationStateOversized(
                "Correlation execution state must be bounded JSON data."
            ) from None
        if len(packet) > self._limits.max_packet_bytes:
            raise CorrelationStateOversized(
                "Correlation execution state exceeds the configured packet bound."
            )
        return packet

    def _now(self) -> float:
        value = self._clock()
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise RuntimeError("Correlation clock returned an invalid value.")
        return float(value)

    def _evict_lru(self) -> None:
        while len(self._entries) > self._limits.max_entries:
            self._entries.popitem(last=False)
        self._condition.notify_all()

    def _purge_expired(self, now: float) -> None:
        expired = [
            key for key, entry in self._entries.items() if now >= entry.expires_at
        ]
        for key in expired:
            del self._entries[key]
        self._purge_generation_reservations(now)
        if expired:
            self._condition.notify_all()

    def _purge_generation_reservations(self, now: float) -> None:
        expired = [
            key
            for key, expires_at in self._generation_reservations.items()
            if now >= expires_at
        ]
        for key in expired:
            del self._generation_reservations[key]
