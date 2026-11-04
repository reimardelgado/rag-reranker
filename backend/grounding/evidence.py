"""Structured evidence packets used by the image-grounding gate.

Evidence content remains untrusted data.  Escaping and canonical validation
belong to the subsequent validation boundary; this module only preserves
references and evaluates the configurable distinct-reference minimum.
"""

from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, Field, field_validator


class EvidenceReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    reference_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    page: str | int

    @field_validator("reference_id", "source")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Reference identifiers and sources cannot be blank.")
        return value

    @field_validator("page")
    @classmethod
    def normalize_page(cls, value: str | int) -> str | int:
        if isinstance(value, bool) or (isinstance(value, str) and not value.strip()):
            raise ValueError("Evidence page must be a non-blank string or integer.")
        return value.strip() if isinstance(value, str) else value

    @property
    def source_page_key(self) -> tuple[str, str]:
        return (self.source.strip().casefold(), str(self.page).strip().casefold())


class EvidenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    reference: EvidenceReference
    content: str = Field(min_length=1)

    @field_validator("content")
    @classmethod
    def reject_blank_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Evidence content cannot be blank.")
        return value


class EvidencePacket(BaseModel):
    """One retrieval result plus its configurable generation threshold."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str = Field(min_length=1)
    items: tuple[EvidenceItem, ...] = ()
    minimum_references: int = Field(default=2, ge=1)

    @field_validator("query")
    @classmethod
    def reject_blank_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Evidence query cannot be blank.")
        return value

    @property
    def references(self) -> tuple[EvidenceReference, ...]:
        seen: set[tuple[str, str]] = set()
        distinct: list[EvidenceReference] = []
        for item in self.items:
            key = item.reference.source_page_key
            if key not in seen:
                seen.add(key)
                distinct.append(item.reference)
        return tuple(distinct)

    @property
    def distinct_reference_count(self) -> int:
        return len(self.references)

    @property
    def is_sufficient(self) -> bool:
        return self.distinct_reference_count >= self.minimum_references


def build_evidence_packet(
    query: str,
    items: Iterable[EvidenceItem | dict[str, object]],
    *,
    minimum_references: int = 2,
) -> EvidencePacket:
    """Build an immutable packet without altering caller-owned evidence."""

    return EvidencePacket(
        query=query,
        items=tuple(EvidenceItem.model_validate(item) for item in items),
        minimum_references=minimum_references,
    )
