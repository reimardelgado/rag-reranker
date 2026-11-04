"""Grounded image instruction contracts and the pre-generation gate.

These pure models are not wired into the production chat route.  Prompt
canonicalization, hashing, claim mapping, and untrusted-evidence escaping are
intentionally reserved for the validation boundary.
"""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .evidence import EvidencePacket, EvidenceReference
from .intent import ImageIntent, classify_image_intent


class DocumentedFact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str = Field(min_length=1)
    reference_ids: tuple[str, ...] = Field(min_length=1)


class GroundedImageInstruction(BaseModel):
    """Structured input to the later canonicalization/validation boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str = Field(min_length=1)
    query: str = Field(min_length=1)
    documented_facts: tuple[DocumentedFact, ...]
    inferred_elements: tuple[str, ...] = ()
    uncertainties: tuple[str, ...] = ()
    forbidden_additions: tuple[str, ...] = ()
    style: str | None = None
    view: str | None = None
    evidence_refs: tuple[EvidenceReference, ...]


class ImageSpec(BaseModel):
    """Validated-image-call shape; validation is implemented in task 7.2."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    canonical_prompt: str = Field(min_length=1)
    size: str | None = None
    quality: str | None = None
    style: str | None = None
    adjacent_label: Literal["Interpretive reconstruction"] = "Interpretive reconstruction"


class GroundingEligibility(str, Enum):
    TEXT = "text"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    ELIGIBLE = "eligible"


def evaluate_grounding_eligibility(
    query: str | None,
    evidence: EvidencePacket | None,
) -> GroundingEligibility:
    """Apply the deterministic intent and evidence gates without side effects."""

    if classify_image_intent(query) is not ImageIntent.EXPLICIT:
        return GroundingEligibility.TEXT
    if evidence is None or not evidence.is_sufficient:
        return GroundingEligibility.INSUFFICIENT_EVIDENCE
    return GroundingEligibility.ELIGIBLE
