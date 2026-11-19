"""Fail-closed validation for canonical grounded image instructions.

This module is deliberately pure and is not wired into the production chat
route.  It treats retrieval content as untrusted JSON data, validates every
documented claim against the supplied evidence packet, and produces a stable
canonical prompt and SHA-256 hash.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
import secrets
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .evidence import EvidencePacket, EvidenceReference
from .instruction import GroundedImageInstruction, ImageSpec
from .render import PROMPT_FORMAT_CANONICAL, InvalidRenderPrompt, render_engine_prompt


class InvalidGroundingError(ValueError):
    """A safe validation failure that must block image-tool emission."""


@dataclass(frozen=True, slots=True)
class GroundingValidationLimits:
    max_canonical_chars: int = 32 * 1024
    max_evidence_chars: int = 256 * 1024
    max_query_chars: int = 4 * 1024
    max_claims: int = 128
    max_claim_chars: int = 2 * 1024
    max_references: int = 128

    def __post_init__(self) -> None:
        if min(
            self.max_canonical_chars,
            self.max_evidence_chars,
            self.max_query_chars,
            self.max_claims,
            self.max_claim_chars,
            self.max_references,
        ) <= 0:
            raise ValueError("Grounding validation bounds must be positive.")


class UntrustedEvidenceBlock(BaseModel):
    """A nonce-delimited block whose body is valid, escaped JSON data."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    nonce: str
    json_data: str
    serialized: str


class ValidatedGrounding(BaseModel):
    """Canonical, bounded output suitable for a later enforcement boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    canonical_json: str
    prompt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    image_spec: ImageSpec
    evidence_block: UntrustedEvidenceBlock


_NONCE_PATTERN = re.compile(r"^[A-Za-z0-9_-]{16,64}$")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _bounded_text(value: str, name: str, maximum: int) -> str:
    normalized = " ".join(value.split())
    if not normalized:
        raise InvalidGroundingError(f"{name} cannot be blank.")
    if len(normalized) > maximum:
        raise InvalidGroundingError(f"{name} exceeds its configured bound.")
    return normalized


def _reference_key(reference: EvidenceReference) -> tuple[str, str, str]:
    return (
        reference.reference_id.strip(),
        reference.source.strip(),
        str(reference.page).strip(),
    )


def _validate_references(
    instruction: GroundedImageInstruction,
    evidence: EvidencePacket,
    limits: GroundingValidationLimits,
) -> dict[str, EvidenceReference]:
    if not evidence.is_sufficient:
        raise InvalidGroundingError("Evidence does not meet the configured minimum.")
    if len(instruction.evidence_refs) > limits.max_references:
        raise InvalidGroundingError("Evidence reference count exceeds its configured bound.")

    packet_by_id: dict[str, EvidenceReference] = {}
    for item in evidence.items:
        reference = item.reference
        reference_id = reference.reference_id.strip()
        existing = packet_by_id.get(reference_id)
        if existing is not None and _reference_key(existing) != _reference_key(reference):
            raise InvalidGroundingError("An evidence reference ID maps to multiple sources.")
        packet_by_id[reference_id] = reference

    canonical_packet_ids = {
        reference.reference_id.strip() for reference in evidence.references
    }

    instruction_by_id: dict[str, EvidenceReference] = {}
    for reference in instruction.evidence_refs:
        reference_id = reference.reference_id.strip()
        if reference_id in instruction_by_id:
            raise InvalidGroundingError("Instruction evidence reference IDs must be unique.")
        packet_reference = packet_by_id.get(reference_id)
        if packet_reference is None or _reference_key(packet_reference) != _reference_key(reference):
            raise InvalidGroundingError("Instruction evidence must map exactly to retrieved evidence.")
        instruction_by_id[reference_id] = reference
    if set(instruction_by_id) != canonical_packet_ids:
        raise InvalidGroundingError("Instruction must include every distinct retrieved reference.")
    return instruction_by_id


def _normalized_list(
    values: tuple[str, ...],
    name: str,
    limits: GroundingValidationLimits,
) -> list[str]:
    if len(values) > limits.max_claims:
        raise InvalidGroundingError(f"{name} count exceeds its configured bound.")
    normalized = [
        _bounded_text(value, f"{name} item", limits.max_claim_chars)
        for value in values
    ]
    return sorted(dict.fromkeys(normalized), key=str.casefold)


def _canonical_instruction(
    instruction: GroundedImageInstruction,
    evidence: EvidencePacket,
    limits: GroundingValidationLimits,
) -> dict[str, Any]:
    instruction_query = _bounded_text(instruction.query, "Query", limits.max_query_chars)
    evidence_query = _bounded_text(evidence.query, "Evidence query", limits.max_query_chars)
    if instruction_query != evidence_query:
        raise InvalidGroundingError("Instruction and evidence queries must match.")

    references = _validate_references(instruction, evidence, limits)
    if not instruction.documented_facts:
        raise InvalidGroundingError("At least one documented fact is required.")
    if len(instruction.documented_facts) > limits.max_claims:
        raise InvalidGroundingError("Documented fact count exceeds its configured bound.")

    facts: list[dict[str, Any]] = []
    documented_text: set[str] = set()
    for fact in instruction.documented_facts:
        text = _bounded_text(fact.text, "Documented fact", limits.max_claim_chars)
        reference_ids = tuple(dict.fromkeys(value.strip() for value in fact.reference_ids))
        if not reference_ids or any(reference_id not in references for reference_id in reference_ids):
            raise InvalidGroundingError("Every documented fact must map to retrieved evidence.")
        entry: dict[str, Any] = {"reference_ids": sorted(reference_ids), "text": text}
        if fact.text_en is not None and fact.text_en.strip():
            entry["text_en"] = _bounded_text(
                fact.text_en, "Documented fact (en)", limits.max_claim_chars
            )
        facts.append(entry)
        documented_text.add(text.casefold())
    facts = sorted(
        {_canonical_json(fact): fact for fact in facts}.values(),
        key=lambda fact: (fact["text"].casefold(), fact["reference_ids"]),
    )

    inferred = _normalized_list(instruction.inferred_elements, "Inferred element", limits)
    uncertainties = _normalized_list(instruction.uncertainties, "Uncertainty", limits)
    forbidden = _normalized_list(instruction.forbidden_additions, "Forbidden addition", limits)
    if documented_text.intersection(value.casefold() for value in forbidden):
        raise InvalidGroundingError("Documented facts cannot also be forbidden additions.")

    canonical: dict[str, Any] = {
        "documented_facts": facts,
        "evidence_refs": [
            {
                "page": reference.page,
                "reference_id": reference.reference_id.strip(),
                "source": reference.source.strip(),
            }
            for reference in sorted(references.values(), key=lambda value: value.reference_id)
        ],
        "forbidden_additions": forbidden,
        "inferred_elements": inferred,
        "query": instruction_query,
        "uncertainties": uncertainties,
        "version": _bounded_text(instruction.version, "Version", 128),
    }
    if instruction.style is not None:
        canonical["style"] = _bounded_text(instruction.style, "Style", limits.max_claim_chars)
    if instruction.view is not None:
        canonical["view"] = _bounded_text(instruction.view, "View", limits.max_claim_chars)
    return canonical


def canonicalize_grounded_instruction(
    instruction: GroundedImageInstruction,
    evidence: EvidencePacket,
    *,
    limits: GroundingValidationLimits | None = None,
) -> str:
    """Return deterministic canonical JSON after all grounding checks pass."""

    resolved_limits = limits or GroundingValidationLimits()
    canonical_json = _canonical_json(
        _canonical_instruction(instruction, evidence, resolved_limits)
    )
    if len(canonical_json.encode("utf-8")) > resolved_limits.max_canonical_chars:
        raise InvalidGroundingError("Canonical instruction exceeds its configured bound.")
    return canonical_json


def hash_canonical_prompt(canonical_prompt: str) -> str:
    """Hash the exact UTF-8 canonical prompt used for a later image call."""

    if not canonical_prompt:
        raise InvalidGroundingError("Canonical prompt cannot be empty.")
    return hashlib.sha256(canonical_prompt.encode("utf-8")).hexdigest()


def serialize_untrusted_evidence(
    evidence: EvidencePacket,
    *,
    nonce: str | None = None,
    limits: GroundingValidationLimits | None = None,
) -> UntrustedEvidenceBlock:
    """Serialize evidence as escaped JSON between unpredictable delimiters.

    Escaping angle brackets in the JSON representation prevents evidence text
    from closing or opening a delimiter while retaining exact data on JSON
    decoding.  The block is data only; callers must never concatenate it into
    policy, tool configuration, or executable instructions.
    """

    resolved_limits = limits or GroundingValidationLimits()
    resolved_nonce = nonce or secrets.token_hex(16)
    if not _NONCE_PATTERN.fullmatch(resolved_nonce):
        raise InvalidGroundingError("Evidence nonce has an invalid format.")

    data = {
        "items": [
            {
                "content": item.content,
                "reference": item.reference.model_dump(mode="json"),
            }
            for item in evidence.items
        ],
        "query": evidence.query,
    }
    json_data = _canonical_json(data).replace("<", r"\u003c").replace(">", r"\u003e")
    if len(json_data.encode("utf-8")) > resolved_limits.max_evidence_chars:
        raise InvalidGroundingError("Serialized evidence exceeds its configured bound.")

    begin = f"<<<UNTRUSTED_EVIDENCE:{resolved_nonce}:BEGIN>>>"
    end = f"<<<UNTRUSTED_EVIDENCE:{resolved_nonce}:END>>>"
    return UntrustedEvidenceBlock(
        nonce=resolved_nonce,
        json_data=json_data,
        serialized=f"{begin}\n{json_data}\n{end}",
    )


def validate_grounded_instruction(
    instruction: GroundedImageInstruction,
    evidence: EvidencePacket,
    *,
    size: str | None = None,
    quality: str | None = None,
    style: str | None = None,
    nonce: str | None = None,
    limits: GroundingValidationLimits | None = None,
    prompt_format: str = PROMPT_FORMAT_CANONICAL,
) -> ValidatedGrounding:
    """Validate, canonicalize, hash, and bound one grounded instruction."""

    resolved_limits = limits or GroundingValidationLimits()
    canonical_json = canonicalize_grounded_instruction(
        instruction,
        evidence,
        limits=resolved_limits,
    )
    prompt_hash = hash_canonical_prompt(canonical_json)
    try:
        render_prompt = render_engine_prompt(canonical_json, prompt_format)
    except InvalidRenderPrompt:
        raise InvalidGroundingError("Engine prompt could not be derived.") from None
    image_spec = ImageSpec(
        canonical_prompt=canonical_json,
        render_prompt=render_prompt,
        size=size,
        quality=quality,
        style=style,
    )
    return ValidatedGrounding(
        canonical_json=canonical_json,
        prompt_hash=prompt_hash,
        image_spec=image_spec,
        evidence_block=serialize_untrusted_evidence(
            evidence,
            nonce=nonce,
            limits=resolved_limits,
        ),
    )
