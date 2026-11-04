# Redis / multi-replica correlation decision

**Change:** `contextual-rag-image-generation` task 8.2  
**Decision:** **NO-GO for multi-replica image-mode rollout**  
**Scope:** decision/spike boundary only; production execution remains inactive

## Evidence from this repository

- `backend/correlation.py` is deliberately process-local and uses Python locks.
- `backend/requirements.txt` and `backend/requirements-dev.txt` contain no Redis
  client.
- `docker-compose.yml` defines no Redis-compatible service.
- No available task-scoped Redis runtime or approved dependency installation exists.

Therefore a real cross-process atomicity spike cannot be executed offline without
adding infrastructure/dependencies or operating a service outside task 8.2's
constraints. Process-local tests MUST NOT be presented as multi-replica proof.

## Verified local contract

The inactive `CorrelationExecutionStore` verifies, within one Python process:

1. one retrieval attempt per exact HMAC-isolated loop and retrieval fingerprint;
2. detached same-loop retrieval reuse, with mismatch, expiry, failure, and size
   bounds failing closed;
3. exactly three atomic tool-iteration claims;
4. one atomic generation reservation per idempotency key and TTL under concurrent
   threads; and
5. no persistence or logging of raw trusted IDs, service/auth values, prompts,
   provider-private signatures, or user data.

This is sufficient for deterministic unit behavior, not production activation.

## Required Redis spike before replica count exceeds one

A future, separately approved spike must use a pinned Redis client and disposable
Redis-compatible service and prove all of the following against at least two worker
processes:

- keys are derived only from opaque HMAC/idempotency digests and have a hard 600
  second TTL;
- retrieval uses an atomic single-flight claim plus bounded immutable result, and
  waiters never repeat retrieval or observe partial state;
- the tool counter uses one atomic script/transaction that returns 1, 2, 3 and
  rejects every later claim;
- paid generation uses an atomic create-if-absent reservation for the full TTL so
  only one worker can win;
- the combined packet is rejected above 256 KiB and capacity/eviction behavior is
  deterministic;
- owner crash, timeout, Redis loss, failover, retry, and cancellation fail closed
  without a second paid call or cross-user reuse; and
- fixtures, logs, errors, and traces contain no secrets, raw IDs, prompts, auth
  values, provider-private signatures, or user data.

Until that evidence exists, deployments with more than one backend replica and all
grounded image runtime activation are **NO-GO**. The safe fallback is the existing
text-only path with image/tool execution disabled.
