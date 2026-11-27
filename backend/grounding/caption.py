"""Deterministic captions and refusal messages for grounded images.

Small local models ignore the caption language and may paraphrase beyond the
evidence.  The template caption is built only from the validated provenance
(documented facts, their references and the stated uncertainties), so it is
always in the configured language and cannot add undocumented content.
"""

from __future__ import annotations

import unicodedata

from .provenance import ImageProvenance

_ENGLISH = {"english", "ingles", "en"}
_MAX_SUBJECT_CHARS = 120
_MAX_FACTS = 6
_MAX_UNCERTAINTIES = 3


def caption_locale(language: str | None) -> str:
    """Map ``GROUNDED_IMAGE_CAPTION_LANGUAGE`` to a display locale.

    The project corpus and its users are Spanish-speaking, so anything other
    than an explicit English setting (including an empty value) uses Spanish.
    """

    if not language:
        return "es"
    folded = unicodedata.normalize("NFKD", language).encode("ascii", "ignore").decode()
    return "en" if folded.strip().casefold() in _ENGLISH else "es"


def _subject(subject: str) -> str:
    text = " ".join(subject.split())
    if len(text) > _MAX_SUBJECT_CHARS:
        text = text[: _MAX_SUBJECT_CHARS - 1].rstrip() + "…"
    return text


def _citation(provenance: ImageProvenance, reference_ids: tuple[str, ...]) -> str:
    by_id = {reference.reference_id: reference for reference in provenance.references}
    cited = []
    for reference_id in reference_ids:
        reference = by_id.get(reference_id)
        if reference is not None:
            cited.append(f"{reference.source} p.{reference.page}")
    return f" [{'; '.join(cited)}]" if cited else ""


def build_template_caption(
    provenance: ImageProvenance, subject: str, locale: str = "es"
) -> str:
    """Caption made only of validated facts, references and uncertainties."""

    english = locale == "en"
    sources = len(provenance.references)
    name = _subject(subject)
    if english:
        lines = [
            f"Interpretive reconstruction of **{name}** based on {sources} "
            "sources from the internal documentation.",
            "",
            "Documented elements:",
        ]
    else:
        lines = [
            f"Reconstrucción interpretativa de **{name}** basada en {sources} "
            "fuentes de la documentación interna.",
            "",
            "Elementos documentados:",
        ]
    for fact in provenance.documented_elements[:_MAX_FACTS]:
        text = (fact.text_en or fact.text) if english else fact.text
        lines.append(f"- {text}{_citation(provenance, fact.reference_ids)}")
    uncertainties = provenance.uncertainties[:_MAX_UNCERTAINTIES]
    if uncertainties:
        lines += ["", "Not documented (not depicted with certainty):" if english
                  else "Sin documentar (no se representa con certeza):"]
        lines += [f"- {item}" for item in uncertainties]
    return "\n".join(lines)


def insufficient_evidence_message(
    subject: str, found: int, required: int, locale: str = "es"
) -> str:
    """User-facing refusal when the corpus lacks enough distinct sources."""

    name = _subject(subject)
    if locale == "en":
        return (
            f"I cannot generate a grounded image of **{name}**: the internal "
            f"documentation provides {found} distinct source(s) on this subject "
            f"and at least {required} are required.\n\n"
            "Try a subject covered by the corpus, rephrase the request, or add "
            "documents about it."
        )
    return (
        f"No puedo generar una imagen fundamentada de **{name}**: la documentación "
        f"interna aporta {found} fuente(s) distinta(s) sobre el tema y se necesitan "
        f"al menos {required}.\n\n"
        "Prueba con un tema que aparezca en el corpus, reformula la petición o "
        "añade documentos sobre él."
    )


def retrieval_unavailable_message(locale: str = "es") -> str:
    if locale == "en":
        return (
            "I cannot generate the image right now: the documentation search is "
            "unavailable. Please try again in a few minutes."
        )
    return (
        "No puedo generar la imagen ahora: la búsqueda en la documentación no está "
        "disponible. Inténtalo de nuevo en unos minutos."
    )
