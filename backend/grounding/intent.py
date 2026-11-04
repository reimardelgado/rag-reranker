"""Deterministic image-intent classification.

Only an explicit image-making verb combined with an image or reconstruction
noun is eligible.  Questions about images and ambiguous visual language stay
on the established text-chat path.
"""

from enum import Enum
import re
import unicodedata


class ImageIntent(str, Enum):
    EXPLICIT = "explicit"
    TEXT = "text"


_IMAGE_VERBS = frozenset(
    {
        "build",
        "create",
        "draw",
        "generate",
        "illustrate",
        "make",
        "produce",
        "reconstruct",
        "render",
        "crea",
        "crear",
        "dibuja",
        "dibujar",
        "genera",
        "generar",
        "haz",
        "ilustra",
        "ilustrar",
        "produce",
        "producir",
        "reconstruir",
        "reconstruye",
        "recrea",
        "recrear",
        "renderiza",
        "renderizar",
    }
)

_IMAGE_NOUNS = frozenset(
    {
        "depiction",
        "illustration",
        "image",
        "picture",
        "reconstruction",
        "render",
        "visualization",
        "imagen",
        "ilustracion",
        "reconstruccion",
        "recreacion",
        "representacion",
        "visualizacion",
    }
)


def _tokens(text: str) -> set[str]:
    normalized = unicodedata.normalize("NFKD", text.casefold())
    without_marks = "".join(char for char in normalized if not unicodedata.combining(char))
    return set(re.findall(r"[^\W_]+", without_marks, flags=re.UNICODE))


def classify_image_intent(query: str | None) -> ImageIntent:
    """Classify only deterministic, explicit image-generation requests.

    Requiring both categories deliberately favors false negatives over
    accidental paid or ungrounded generation.  Empty/attachment-only turns,
    image discussion, and ambiguous requests therefore remain text.
    """

    if not isinstance(query, str) or not query.strip():
        return ImageIntent.TEXT
    tokens = _tokens(query)
    if tokens & _IMAGE_VERBS and tokens & _IMAGE_NOUNS:
        return ImageIntent.EXPLICIT
    return ImageIntent.TEXT
