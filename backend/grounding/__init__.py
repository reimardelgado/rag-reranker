"""Pure, inactive grounding contracts for evidence-backed image requests."""

from .evidence import EvidenceItem, EvidencePacket, EvidenceReference, build_evidence_packet
from .instruction import (
    DocumentedFact,
    GroundedImageInstruction,
    GroundingEligibility,
    ImageSpec,
    evaluate_grounding_eligibility,
)
from .intent import ImageIntent, classify_image_intent
from .provenance import (
    ADJACENT_LABEL,
    DOCUMENTARY_DISCLAIMER,
    DocumentedElement,
    ImageProvenance,
    InvalidProvenanceError,
    ProvenanceReference,
    build_image_provenance,
    render_adjacent_provenance,
)
from .validation import (
    GroundingValidationLimits,
    InvalidGroundingError,
    UntrustedEvidenceBlock,
    ValidatedGrounding,
    canonicalize_grounded_instruction,
    hash_canonical_prompt,
    serialize_untrusted_evidence,
    validate_grounded_instruction,
)

__all__ = [
    "DocumentedFact",
    "DocumentedElement",
    "ADJACENT_LABEL",
    "DOCUMENTARY_DISCLAIMER",
    "EvidenceItem",
    "EvidencePacket",
    "EvidenceReference",
    "GroundedImageInstruction",
    "GroundingEligibility",
    "GroundingValidationLimits",
    "ImageIntent",
    "ImageSpec",
    "ImageProvenance",
    "InvalidGroundingError",
    "InvalidProvenanceError",
    "ProvenanceReference",
    "UntrustedEvidenceBlock",
    "ValidatedGrounding",
    "build_evidence_packet",
    "build_image_provenance",
    "canonicalize_grounded_instruction",
    "classify_image_intent",
    "evaluate_grounding_eligibility",
    "hash_canonical_prompt",
    "serialize_untrusted_evidence",
    "render_adjacent_provenance",
    "validate_grounded_instruction",
]
