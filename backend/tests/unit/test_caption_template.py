from __future__ import annotations

import pytest

from backend.grounding.caption import (
    build_template_caption,
    caption_locale,
    insufficient_evidence_message,
    retrieval_unavailable_message,
)
from backend.grounding.provenance import (
    ADJACENT_LABEL,
    PROVENANCE_LABELS,
    ImageProvenance,
    render_adjacent_provenance,
)


def _provenance(**overrides) -> ImageProvenance:
    values = dict(
        prompt_version="grounded-image-v1",
        prompt_hash="a" * 64,
        references=(
            {"reference_id": "ref-1", "source": "Jaen_tierra_ibera.pdf", "page": 511},
            {"reference_id": "ref-2", "source": "Giribaile.pdf", "page": "12"},
        ),
        documented_elements=(
            {"text": "Tiene una fortificación de tipo barrera.", "text_en": "It has a barrier wall.",
             "reference_ids": ("ref-1",)},
            {"text": "Está en Vilches, Jaén.", "reference_ids": ("ref-1", "ref-2", "ref-missing")},
        ),
        inferred_elements=(),
        uncertainties=("Número de habitantes",),
        forbidden_items=(),
        retrieval_model_id="embedder",
        chat_model_id="qwen2.5:3b",
        image_model_id="sd15",
        timestamp="2026-11-26T10:00:00Z",
    )
    values.update(overrides)
    return ImageProvenance(**values)


@pytest.mark.parametrize(
    ("language", "locale"),
    [(None, "es"), ("", "es"), ("Spanish", "es"), ("Español", "es"),
     ("English", "en"), ("inglés", "en"), ("en", "en"), ("French", "es")],
)
def test_caption_locale(language, locale):
    assert caption_locale(language) == locale


def test_template_caption_uses_only_validated_facts_and_cites_references():
    caption = build_template_caption(_provenance(), "oppidum de Giribaile")
    assert caption.startswith(
        "Reconstrucción interpretativa de **oppidum de Giribaile** basada en 2 fuentes"
    )
    assert "- Tiene una fortificación de tipo barrera. [Jaen_tierra_ibera.pdf p.511]" in caption
    # Unknown reference IDs are skipped, never invented.
    assert "- Está en Vilches, Jaén. [Jaen_tierra_ibera.pdf p.511; Giribaile.pdf p.12]" in caption
    assert "Sin documentar (no se representa con certeza):\n- Número de habitantes" in caption


def test_template_caption_omits_uncertainty_block_when_empty_and_supports_english():
    caption = build_template_caption(_provenance(uncertainties=()), "x", "en")
    assert caption.startswith("Interpretive reconstruction of **x**")
    assert "- It has a barrier wall." in caption
    assert "- Está en Vilches, Jaén." in caption  # falls back to text without text_en
    assert "Not documented" not in caption


def test_long_subject_is_truncated():
    caption = build_template_caption(_provenance(), "a" * 500)
    assert "a" * 119 + "…" in caption and "a" * 120 not in caption


def test_refusal_messages():
    message = insufficient_evidence_message("astronauta en Marte", 0, 2)
    assert message.startswith("No puedo generar una imagen fundamentada de **astronauta en Marte**")
    assert "0 fuente(s)" in message and "al menos 2" in message
    assert "cannot" in insufficient_evidence_message("x", 1, 2, "en")
    assert retrieval_unavailable_message().startswith("No puedo generar la imagen ahora")


def test_provenance_labels_follow_locale_and_default_to_english():
    spanish = render_adjacent_provenance(_provenance(), locale="es")
    assert PROVENANCE_LABELS["es"]["label"] in spanish and ADJACENT_LABEL not in spanish
    assert "<summary>Procedencia</summary>" in spanish
    assert ADJACENT_LABEL in render_adjacent_provenance(_provenance())
