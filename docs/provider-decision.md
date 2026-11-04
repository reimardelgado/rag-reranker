# Provider Architecture Decision

## Decision record

| Field | Value |
|---|---|
| Decision | **GO — implement a native Gemini adapter inside the RAG backend** |
| Status | Approved architecture; implementation remains gated to tasks 6.1 and 6.2 |
| Date | 2026-10-30 |
| Approver context | Explicit user/architect approval after review of the completed task 5.1 and 5.2 runtime evidence |
| Client-facing protocol | OpenAI-compatible chat tools and SSE toward Open WebUI |
| Provider-facing protocol | Native Gemini function-calling and continuation protocol |

This decision completes SDD task 5.3. It authorizes only the provider adapter work
defined by tasks 6.1 and 6.2; it does not enable grounded image generation or make
an image call.

## Evidence reviewed

All evidence is sanitized and contains no credential values, raw prompts, raw
provider output, tool arguments, opaque IDs, or thought-signature values.

### Open WebUI v0.11.4 — task 5.1

Source: `backend/tests/fixtures/openwebui/v0.11.4-runtime-capture.json`

- The Open WebUI-to-RAG request is wire-proven to use the Native tool-schema path.
- The request carried 34 tools; the redaction allowlist identified
  `generate_image` exactly once and intentionally withheld the other tool names.
- `stream` was present and `true`.
- `tool_choice`, `stream_options`, `options`, `params`, and `features` were absent
  from the backend-bound request.
- The deployed bridge discarded tools at its narrow request/provider boundary,
  explaining the historical text-only result despite Open WebUI injection.
- No trusted user/chat/root-message correlation headers reached the backend. Only
  header names were captured; no values were persisted.
- Browser-to-Open WebUI body/header evidence remains deliberately security-blocked.
  This does not weaken the wire proof for the Open WebUI-to-RAG hop.

### Gemini chat conformance — task 5.2

Source: `backend/tests/fixtures/providers/gemini-openai-chat-conformance.json`

- The OpenAI-compatible non-stream function call passed: one allowlisted function,
  a string ID, valid schema-conforming JSON arguments, and
  `finish_reason=tool_calls`. A thought signature was structurally present in the
  provider-specific extra-content location.
- Direct OpenAI-compatible streaming, tool-result continuation, and multimodal
  fidelity were not qualified. The direct compatibility path therefore did not
  pass the production gate.
- Native Gemini function calling passed. A native thought signature was present,
  and the exact signature/content was held only in memory and accepted on the
  matching native function-response continuation. The continuation completed
  successfully without another function call.

## Selected architecture and boundaries

The backend will be the protocol adapter, not the image executor:

1. **Open WebUI** owns image execution, image-provider credentials, generated
   files, RBAC/capability checks, and UI events. It sends OpenAI-compatible chat
   requests and consumes OpenAI-compatible non-stream or SSE responses.
2. **RAG backend API boundary** preserves messages, tools, tool choice when
   supplied, multimodal parts, and supported unknown fields. Grounding,
   validation, trusted correlation, idempotency, and fail-closed enforcement stay
   in the backend's existing design boundaries.
3. **Native Gemini adapter inside the backend** translates preserved chat state
   and function declarations to Gemini's native protocol, retains provider-private
   continuation metadata in memory, and translates native function calls and text
   back into normalized OpenAI-compatible events.
4. **Gemini provider** sees only the native Gemini request/continuation protocol.
   Open WebUI never needs to understand Gemini thought signatures.

The adapter must conform to the existing `ProviderAdapter.complete(request) ->
AsyncIterator[NormalizedEvent]` design boundary. This decision does not move image
ownership into the backend.

## Rejected production paths

### Direct OpenAI-compatible Gemini passthrough — NO-GO

Direct passthrough is not production-safe because only non-stream function calling
was proven. Streamed tool-call indexes/identity/argument assembly and `[DONE]`
semantics, OpenAI-compatible continuation, multimodal fidelity, and signature
round-trip were not qualified. Open WebUI requires a conformant streaming tool loop,
and Gemini requires exact thought-signature handling on continuation. Passing the
provider response through unchanged would make those correctness properties
accidental rather than enforced.

### Gateway — fallback only

A gateway is not the selected primary architecture. It may be reconsidered only if
the native adapter cannot satisfy the task 6.1/6.2 acceptance gate and only after
the candidate gateway proves the same stream, continuation, multimodal, signature,
redaction, and failure contracts. A gateway adds another trust, availability,
latency, configuration, and observability boundary, so convenience alone is not a
reason to adopt it.

### Sequential image-instruction tool fallback — not selected

The sequential-tool approach remains an emergency design fallback, not an
implementation target. It adds a model/tool round trip, permits mutation of the
canonical instruction, increases latency, and has no completed sequential-tool
runtime proof. It requires a separate spike and architecture approval before use.

## Thought-signature handling

Thought signatures are opaque provider continuation metadata. The adapter must:

- capture the exact signature associated with the native model function-call part;
- keep it only in bounded in-memory continuation state and return it unchanged with
  the matching native function response;
- never serialize it into OpenAI tool arguments, client-visible fields,
  provenance, metrics, exceptions, traces, or application logs;
- never persist it to a database, fixture, cache backing store, or disk;
- never derive, inspect, transform, truncate, hash for observability, or invent a
  replacement signature; and
- discard it on completion, expiry, mismatch, cancellation, or error.

If an exact required signature cannot be recovered for the matching continuation,
the adapter must reject the continuation with a recoverable error and must not send
an unsigned or guessed provider request.

## Outbound OpenAI-compatible normalization

The backend must normalize native Gemini output rather than expose native payloads.

- Streaming responses use `text/event-stream` OpenAI chat-completion chunks.
- Every tool call has one stable response-local index and ID. A valid provider
  identity may be retained; otherwise the adapter assigns one without exposing a
  provider signature or raw identifier.
- The function name is emitted once, arguments are emitted as ordered incremental
  JSON string fragments, and the assembled arguments must validate before release
  to an executable image-tool path.
- The terminal tool chunk has `finish_reason=tool_calls`, followed by exactly one
  `[DONE]` sentinel.
- Text deltas and text finish reasons preserve their semantics without leaking
  native provider envelopes.
- Non-stream responses are protocol-equivalent: assistant `tool_calls` carry the
  same stable identity/name/arguments and `finish_reason=tool_calls`.
- A later `role=tool` message must match the prior tool-call identity and produce a
  valid native function-response continuation, then a normalized final assistant
  response.
- Missing indexes, malformed arguments, unsupported names, invalid signatures,
  incompatible payloads, timeouts, throttling, and provider 5xx responses become
  bounded OpenAI-shaped recoverable errors. Explicit image intent fails closed.

## Security and operational constraints

- **Trusted correlation:** body-supplied IDs are never trusted. The current runtime
  has no trusted correlation headers, so continuation/image enablement remains
  blocked until service authentication plus trusted user/session/root-message
  correlation is configured and validated on a private boundary.
- **Isolation and bounds:** continuation state is keyed by an HMAC of trusted
  identity/context, isolated per user/conversation, TTL/size/entry bounded, and
  rejected on mismatch, expiry, or oversize. The design limits are 10 minutes,
  256 KiB per evidence packet, a configured entry cap, and three tool iterations.
- **Idempotency:** use the designed SHA-256 identity/context/prompt-version/prompt-
  hash key and suppress duplicates so retries cannot cause more than one paid image
  call per TTL. Provider call IDs and thought signatures are not idempotency keys.
- **Fail closed:** absent evidence, invalid grounding, unsupported/missing
  `generate_image`, invalid tool arguments, untrusted correlation, state failure,
  loop overflow, provider incompatibility, or normalization failure emits no
  unconstrained image call.
- **Redaction:** logs, metrics, traces, errors, fixtures, and provenance exclude
  credentials, authorization values, raw prompts/evidence/provider payloads, tool
  arguments, opaque IDs/signatures, and unnecessary user data. Use bounded status,
  category, count, timing, and approved hash fields only.
- **Preservation:** incoming multimodal parts and supported fields remain lossless at
  the API boundary; retrieval derives text without mutating forwarded messages.
- **Access/cost:** Open WebUI remains responsible for image permissions and secrets;
  backend feature flags and allowlists default off. No paid action occurs on denial,
  rollback, or validation failure.

## Implementation gate for tasks 6.1 and 6.2

Task 6.1 may implement only the selected native Gemini adapter. Task 6.2 may wire
only protocol normalization into the existing chat boundary. Neither task enables
grounded image execution.

Acceptance requires deterministic offline tests proving all of the following:

1. Native request translation preserves roles, tools, supported tool choice,
   assistant function calls, matching function responses, multimodal parts, and
   supported unknown fields without mutating the input contract.
2. Native function-call arguments normalize to valid OpenAI tool calls in both
   stream and non-stream modes with stable index/ID/name, ordered argument
   assembly, `finish_reason=tool_calls`, and exactly one `[DONE]` in SSE mode.
3. Exact thought-signature continuation succeeds in memory, while missing,
   mismatched, expired, or replayed state is rejected without provider dispatch.
4. No thought signature, opaque provider identifier, raw prompt/evidence/arguments,
   credential, or authorization value is logged, persisted, exposed to Open WebUI,
   or included in errors/metrics.
5. Text-only stream and non-stream behavior remains compatible, and multimodal
   content is forwarded faithfully rather than flattened.
6. Malformed native payloads, invalid/unknown tools, timeout, 429, 5xx, cancellation,
   and loop overflow produce bounded recoverable OpenAI error behavior and never an
   unconstrained image tool call.
7. Focused adapter/normalization contracts, the full current test suite, fixture
   sanitization checks, and `git diff --check` pass. Live or paid image calls are
   excluded.

Failure of any acceptance item is **NO-GO for image-mode rollout**. It triggers
text-only operation while the native adapter is corrected or a separately validated
gateway fallback is proposed for approval.

## Rollback

Rollback is text-only mode: disable the grounded-image/tool capability and native
adapter path with the default-off feature flag, retain the existing text chat
contract, and emit no `generate_image` call. Open WebUI image execution settings may
remain independently configured, but the RAG backend must not advertise or release
an image tool call until the full later rollout gates pass. No migration of image
files or backend-held image credentials is required because the backend owns
neither.

## Explicit non-goals of this decision

- No backend ownership of image generation, image-provider credentials, binary
  storage, file authorization, or image UI events.
- No guaranteed in-pixel watermark; the design provides an adjacent
  **Interpretive reconstruction** label and sidecar provenance later.
- No image generation request, paid smoke call, image-provider validation, or
  runtime rollout as part of this decision.
- No implementation of tasks 6.1, 6.2, grounding, correlation, enforcement, or
  provenance in this record.
