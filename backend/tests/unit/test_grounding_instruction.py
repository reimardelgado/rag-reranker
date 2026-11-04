from copy import deepcopy

import pytest
from pydantic import ValidationError

from backend.grounding.evidence import build_evidence_packet
from backend.grounding.instruction import (
    GroundedImageInstruction,
    GroundingEligibility,
    ImageSpec,
    evaluate_grounding_eligibility,
)
from backend.grounding.intent import ImageIntent, classify_image_intent


def _items():
    return [
        {
            "reference": {"reference_id": "ref-a", "source": "Survey.pdf", "page": 12},
            "content": "The western wall used large stone blocks.",
        },
        {
            "reference": {"reference_id": "ref-b", "source": "Excavation.pdf", "page": "8"},
            "content": "The entrance opened toward the south.",
        },
    ]


@pytest.mark.parametrize(
    "query",
    [
        "Generate an image of the documented settlement",
        "Crea una reconstrucción del asentamiento documentado",
        "RENDERIZA una IMAGEN del oppidum",
    ],
)
def test_explicit_image_intent_requires_a_generation_verb_and_visual_noun(query):
    assert classify_image_intent(query) is ImageIntent.EXPLICIT


@pytest.mark.parametrize(
    "query",
    [
        "What does the excavation image show?",
        "Could this have looked like a fortified settlement?",
        "Reconstruye el asentamiento",
        "Create a detailed description of the settlement",
        "",
        None,
    ],
)
def test_ambiguous_or_incomplete_intent_stays_text(query):
    assert classify_image_intent(query) is ImageIntent.TEXT


def test_evidence_gate_counts_distinct_source_page_pairs_without_mutation():
    items = _items()
    items.append(
        {
            "reference": {"reference_id": "duplicate-id", "source": " survey.PDF ", "page": 12},
            "content": "A second chunk from the same page.",
        }
    )
    original = deepcopy(items)

    packet = build_evidence_packet("create an image", items)

    assert packet.minimum_references == 2
    assert packet.distinct_reference_count == 2
    assert packet.is_sufficient is True
    assert [reference.reference_id for reference in packet.references] == ["ref-a", "ref-b"]
    assert items == original


def test_reference_minimum_is_configurable_and_must_be_positive():
    packet = build_evidence_packet("create an image", _items(), minimum_references=3)

    assert packet.is_sufficient is False
    assert evaluate_grounding_eligibility(packet.query, packet) is GroundingEligibility.INSUFFICIENT_EVIDENCE
    with pytest.raises(ValidationError):
        build_evidence_packet("create an image", _items(), minimum_references=0)


def test_eligibility_keeps_text_intent_out_and_requires_sufficient_evidence():
    sufficient = build_evidence_packet("Generate an image of the site", _items())
    insufficient = build_evidence_packet("Generate an image of the site", _items()[:1])

    assert evaluate_grounding_eligibility("Explain the site", sufficient) is GroundingEligibility.TEXT
    assert evaluate_grounding_eligibility(sufficient.query, sufficient) is GroundingEligibility.ELIGIBLE
    assert (
        evaluate_grounding_eligibility(insufficient.query, insufficient)
        is GroundingEligibility.INSUFFICIENT_EVIDENCE
    )


def test_instruction_and_image_spec_preserve_structured_constraints():
    packet = build_evidence_packet("Generate an image of the site", _items())
    instruction = GroundedImageInstruction.model_validate(
        {
            "version": "grounded-image-v1",
            "query": packet.query,
            "documented_facts": [
                {"text": "The entrance faced south.", "reference_ids": ["ref-b"]}
            ],
            "inferred_elements": ["roof material"],
            "uncertainties": ["building height"],
            "forbidden_additions": ["modern vehicles"],
            "style": "archaeological reconstruction",
            "view": "ground level",
            "evidence_refs": [reference.model_dump() for reference in packet.references],
        }
    )
    spec = ImageSpec(
        canonical_prompt="Bounded prompt awaiting task 7.2 validation",
        size="1024x1024",
        quality="standard",
        style="natural",
    )

    assert instruction.documented_facts[0].reference_ids == ("ref-b",)
    assert instruction.evidence_refs == packet.references
    assert spec.adjacent_label == "Interpretive reconstruction"
    with pytest.raises(ValidationError):
        ImageSpec(canonical_prompt="prompt", adjacent_label="Documentary image")
