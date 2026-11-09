"""Bounded adjacent provenance for grounded image reconstructions.

The sidecar is derived only from the validated canonical instruction.  Raw
evidence, correlation identifiers, authentication values, and provider-private
continuation metadata are therefore outside this module's input contract.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .render import PROMPT_FORMAT_DIFFUSION, InvalidRenderPrompt, render_engine_prompt
from .validation import ValidatedGrounding, hash_canonical_prompt


ADJACENT_LABEL = "Interpretive reconstruction"
DOCUMENTARY_DISCLAIMER = "Generated pixels are not documentary evidence."


class InvalidProvenanceError(ValueError):
    """Validated grounding could not form a safe bounded sidecar."""


class ProvenanceReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    reference_id: str = Field(min_length=1, max_length=512)
    source: str = Field(min_length=1, max_length=2048)
    page: str | int


class DocumentedElement(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str = Field(min_length=1, max_length=2048)
    reference_ids: tuple[str, ...] = Field(min_length=1, max_length=128)
    text_en: str | None = Field(default=None, min_length=1, max_length=2048)


class ImageProvenance(BaseModel):
    """Public sidecar data; deliberately excludes raw evidence and secrets."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt_version: str = Field(min_length=1, max_length=128)
    prompt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    references: tuple[ProvenanceReference, ...] = Field(max_length=128)
    documented_elements: tuple[DocumentedElement, ...] = Field(max_length=128)
    inferred_elements: tuple[str, ...] = Field(max_length=128)
    uncertainties: tuple[str, ...] = Field(max_length=128)
    forbidden_items: tuple[str, ...] = Field(max_length=128)
    retrieval_model_id: str = Field(min_length=1, max_length=256)
    chat_model_id: str = Field(min_length=1, max_length=256)
    image_model_id: str = Field(min_length=1, max_length=256)
    timestamp: str = Field(min_length=1, max_length=64)
    # Exact prompt the image engine received when it differs from the
    # canonical instruction (diffusion engines); re-derived, never trusted.
    engine_prompt: str | None = Field(default=None, min_length=1, max_length=4096)

    @field_validator(
        "prompt_version",
        "retrieval_model_id",
        "chat_model_id",
        "image_model_id",
        "timestamp",
    )
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Provenance text fields cannot be blank.")
        return value


def _require_list(value: object, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise InvalidProvenanceError(f"Canonical {name} must be a list.")
    return value


def _string_tuple(value: object, name: str) -> tuple[str, ...]:
    items = _require_list(value, name)
    if any(not isinstance(item, str) or not item.strip() for item in items):
        raise InvalidProvenanceError(f"Canonical {name} must contain non-blank text.")
    return tuple(items)


def _public_timestamp(generated_at: datetime) -> str:
    if generated_at.tzinfo is None or generated_at.utcoffset() is None:
        raise InvalidProvenanceError("Provenance timestamp must include a timezone.")
    return (
        generated_at.astimezone(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def build_image_provenance(
    grounding: ValidatedGrounding,
    *,
    retrieval_model_id: str,
    chat_model_id: str,
    image_model_id: str,
    generated_at: datetime,
) -> ImageProvenance:
    """Build provenance from the exact validated prompt that was enforced."""

    if hash_canonical_prompt(grounding.canonical_json) != grounding.prompt_hash:
        raise InvalidProvenanceError("Validated prompt hash does not match its prompt.")
    if grounding.image_spec.canonical_prompt != grounding.canonical_json:
        raise InvalidProvenanceError("Image prompt does not match validated grounding.")
    engine_prompt = grounding.image_spec.render_prompt
    if engine_prompt is not None:
        try:
            expected = render_engine_prompt(grounding.canonical_json, PROMPT_FORMAT_DIFFUSION)
        except InvalidRenderPrompt:
            raise InvalidProvenanceError("Engine prompt cannot be re-derived.") from None
        if engine_prompt != expected:
            raise InvalidProvenanceError("Engine prompt does not match validated grounding.")
    try:
        canonical = json.loads(grounding.canonical_json)
    except (TypeError, json.JSONDecodeError):
        raise InvalidProvenanceError("Validated prompt is not canonical JSON.") from None
    if not isinstance(canonical, dict):
        raise InvalidProvenanceError("Validated prompt must be a JSON object.")
    if json.dumps(
        canonical, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ) != grounding.canonical_json:
        raise InvalidProvenanceError("Validated prompt is not canonical JSON.")

    try:
        references = tuple(
            ProvenanceReference.model_validate(reference)
            for reference in _require_list(canonical.get("evidence_refs"), "references")
        )
        documented = tuple(
            DocumentedElement.model_validate(fact)
            for fact in _require_list(canonical.get("documented_facts"), "facts")
        )
        inferred = _string_tuple(canonical.get("inferred_elements"), "inferences")
        uncertainties = _string_tuple(canonical.get("uncertainties"), "uncertainties")
        forbidden = _string_tuple(canonical.get("forbidden_additions"), "forbidden items")
        version = canonical["version"]
    except (KeyError, TypeError, ValueError) as error:
        raise InvalidProvenanceError("Validated prompt has an invalid provenance shape.") from error

    try:
        return ImageProvenance(
            prompt_version=version,
            prompt_hash=grounding.prompt_hash,
            references=references,
            documented_elements=documented,
            inferred_elements=inferred,
            uncertainties=uncertainties,
            forbidden_items=forbidden,
            retrieval_model_id=retrieval_model_id,
            chat_model_id=chat_model_id,
            image_model_id=image_model_id,
            timestamp=_public_timestamp(generated_at),
            engine_prompt=engine_prompt,
        )
    except InvalidProvenanceError:
        raise
    except ValueError as error:
        raise InvalidProvenanceError("Provenance exceeded its public contract.") from error


def render_adjacent_provenance(
    provenance: ImageProvenance,
    *,
    max_sidecar_chars: int = 64 * 1024,
) -> str:
    """Render an injection-safe collapsible sidecar adjacent to the result."""

    if max_sidecar_chars <= 0:
        raise ValueError("Provenance output bound must be positive.")
    sidecar = json.dumps(
        provenance.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    # References and model labels are display data.  Keep them valid JSON while
    # preventing copied content from closing the details/code containers.
    sidecar = (
        sidecar.replace("&", r"\u0026")
        .replace("<", r"\u003c")
        .replace(">", r"\u003e")
        .replace("`", r"\u0060")
    )
    if len(sidecar.encode("utf-8")) > max_sidecar_chars:
        raise InvalidProvenanceError("Provenance sidecar exceeded its configured bound.")
    return (
        f"\n\n**{ADJACENT_LABEL}**\n\n"
        f"{DOCUMENTARY_DISCLAIMER}\n\n"
        "<details><summary>Provenance</summary>\n\n"
        f"```json\n{sidecar}\n```\n"
        "</details>"
    )
