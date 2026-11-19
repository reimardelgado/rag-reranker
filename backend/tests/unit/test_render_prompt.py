import json
from datetime import datetime, timezone

import pytest

from backend.config import ConfigurationError, GroundedImageConfig
from backend.grounding.evidence import build_evidence_packet
from backend.grounding.instruction import DocumentedFact, GroundedImageInstruction
from backend.grounding.provenance import InvalidProvenanceError, build_image_provenance
from backend.grounding.render import (
    DIFFUSION_PREFIX,
    InvalidRenderPrompt,
    render_diffusion_prompt,
    render_engine_prompt,
)
from backend.grounding.validation import validate_grounded_instruction


def _canonical(facts, **extra):
    return json.dumps({"documented_facts": facts, **extra}, ensure_ascii=False)


def test_diffusion_prompt_uses_only_documented_facts_in_english():
    canonical = _canonical(
        [
            {"text": "Muralla de mampostería", "text_en": "irregular stone masonry wall", "reference_ids": ["r1"]},
            {"text": "Puerta monumental", "text_en": "monumental gate with two towers.", "reference_ids": ["r2"]},
        ],
        inferred_elements=["clear sky"],
        forbidden_additions=["medieval battlements"],
        uncertainties=["exact height"],
        style="watercolor",
    )

    prompt = render_diffusion_prompt(canonical)

    assert prompt.startswith(DIFFUSION_PREFIX + ", watercolor, irregular stone masonry wall")
    assert "monumental gate with two towers," in prompt
    for absent in ("clear sky", "battlements", "exact height", "Muralla", "{", "ref"):
        assert absent not in prompt


def test_falls_back_to_original_text_and_strips_figure_prefixes():
    canonical = _canonical(
        [
            {"text": "Figura 14.- Muralla republicana sobre la muralla ibera", "reference_ids": ["r"]},
            {"text": "Figura 14.- Muralla republicana sobre la muralla ibera", "reference_ids": ["s"]},
        ]
    )
    prompt = render_diffusion_prompt(canonical)
    assert "Figura" not in prompt
    assert prompt.count("Muralla republicana") == 1


def test_prompt_is_bounded_and_deterministic():
    facts = [{"text": f"fact {i}", "text_en": " ".join(["word"] * 20), "reference_ids": ["r"]} for i in range(10)]
    canonical = _canonical(facts)
    first = render_diffusion_prompt(canonical)
    assert first == render_diffusion_prompt(canonical)
    assert len(first.split()) <= 60
    long_single = _canonical([{"text": "x", "text_en": " ".join(["stone"] * 200), "reference_ids": ["r"]}])
    assert 0 < len(render_diffusion_prompt(long_single).split()) <= 60


@pytest.mark.parametrize("bad", ["not json", "[]", _canonical([])])
def test_invalid_canonical_is_rejected(bad):
    with pytest.raises(InvalidRenderPrompt):
        render_diffusion_prompt(bad)


def test_canonical_format_keeps_json_and_unknown_format_fails():
    assert render_engine_prompt(_canonical([{"text": "a", "reference_ids": ["r"]}]), "canonical_json") is None
    with pytest.raises(InvalidRenderPrompt):
        render_engine_prompt("{}", "other")


ITEMS = [
    {
        "reference": {"reference_id": "ref-a", "source": "Jaen.pdf", "page": 12},
        "content": "El oppidum ibero estaba rodeado por una muralla de mampostería.",
    },
    {
        "reference": {"reference_id": "ref-b", "source": "Jaen.pdf", "page": 40},
        "content": "Las viviendas del oppidum ibero eran rectangulares con zócalo de piedra.",
    },
]


def _grounding(prompt_format):
    packet = build_evidence_packet("Genera una imagen del oppidum ibero", ITEMS)
    refs = packet.references
    instruction = GroundedImageInstruction(
        version="grounded-image-v1",
        query=packet.query,
        documented_facts=(
            DocumentedFact(
                text="Muralla de mampostería",
                text_en="stone masonry wall around the oppidum",
                reference_ids=(refs[0].reference_id,),
            ),
            DocumentedFact(text="Viviendas rectangulares", reference_ids=(refs[1].reference_id,)),
        ),
        forbidden_additions=("castillo medieval",),
        evidence_refs=refs,
    )
    return validate_grounded_instruction(instruction, packet, prompt_format=prompt_format)


def _provenance(grounding):
    return build_image_provenance(
        grounding,
        retrieval_model_id="emb",
        chat_model_id="qwen2.5:3b",
        image_model_id="comfyui-sd15",
        generated_at=datetime(2026, 11, 9, tzinfo=timezone.utc),
    )


def test_diffusion_grounding_hashes_text_en_and_records_engine_prompt():
    grounding = _grounding("diffusion")
    assert '"text_en":"stone masonry wall around the oppidum"' in grounding.canonical_json
    assert grounding.image_spec.render_prompt.startswith(DIFFUSION_PREFIX)
    assert "stone masonry wall" in grounding.image_spec.render_prompt
    assert "castillo" not in grounding.image_spec.render_prompt

    provenance = _provenance(grounding)
    assert provenance.engine_prompt == grounding.image_spec.render_prompt
    assert any(element.text_en for element in provenance.documented_elements)


def test_canonical_grounding_has_no_engine_prompt():
    grounding = _grounding("canonical_json")
    assert grounding.image_spec.render_prompt is None
    assert _provenance(grounding).engine_prompt is None


def test_tampered_engine_prompt_fails_provenance():
    grounding = _grounding("diffusion")
    tampered = grounding.model_copy(
        update={"image_spec": grounding.image_spec.model_copy(update={"render_prompt": "a castle"})}
    )
    with pytest.raises(InvalidProvenanceError):
        _provenance(tampered)


def test_prompt_format_defaults_follow_the_mode():
    assert GroundedImageConfig.from_environ({}).prompt_format == "diffusion"
    assert GroundedImageConfig.from_environ({"DEFAULT_PROVIDER": "gemini"}).prompt_format == "canonical_json"
    explicit = GroundedImageConfig.from_environ({"GROUNDED_IMAGE_PROMPT_FORMAT": "canonical_json"})
    assert explicit.prompt_format == "canonical_json"
    with pytest.raises(ConfigurationError):
        GroundedImageConfig.from_environ({"GROUNDED_IMAGE_PROMPT_FORMAT": "markdown"})
