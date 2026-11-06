"""Convert existing hybrid-retrieval text blocks into structured evidence.

The legacy retrieval path renders each chunk as ``"[source p.page]\\ncontent"``
(optionally with a ``[Fase N]`` tag).  The image gate needs typed
``EvidenceItem`` values with stable reference IDs instead.  Blocks whose
header cannot be parsed are dropped rather than guessed, so an unparseable
block can never count towards the evidence minimum.
"""

from __future__ import annotations

from collections.abc import Iterable
import hashlib
import math
import re
import unicodedata

from .evidence import EvidenceItem, EvidencePacket, EvidenceReference, build_evidence_packet
from .intent import _IMAGE_NOUNS, _IMAGE_VERBS


_BLOCK = re.compile(
    r"^\[(?P<source>[^\]\n]+?) p\.(?P<page>[^\]\s]+)(?: \[Fase [^\]\n]*\])?\]\n(?P<content>.+)$",
    re.DOTALL,
)

# Function words that only frame the request ("genera una imagen de ...").
_FRAMING_WORDS = frozenset(
    {
        "a",
        "an",
        "de",
        "del",
        "el",
        "la",
        "las",
        "los",
        "me",
        "of",
        "please",
        "por",
        "favor",
        "the",
        "un",
        "una",
        "unos",
        "unas",
    }
)


def _fold(token: str) -> str:
    normalized = unicodedata.normalize("NFKD", token.casefold())
    return "".join(char for char in normalized if not unicodedata.combining(char))


def retrieval_query_for_image(query: str) -> str:
    """Remove the leading image-request framing so retrieval targets the subject.

    Only the leading run of image verbs/nouns and framing words is removed;
    the subject itself is never altered.  If nothing would remain, the
    original query is returned unchanged.
    """

    tokens = query.split()
    index = 0
    while index < len(tokens):
        folded = _fold(re.sub(r"[^\w]", "", tokens[index]))
        if folded in _IMAGE_VERBS or folded in _IMAGE_NOUNS or folded in _FRAMING_WORDS:
            index += 1
            continue
        break
    remainder = " ".join(tokens[index:]).strip()
    return remainder or query.strip()


def reference_id_for(source: str, page: str) -> str:
    """Stable, opaque ID for one normalized source/page pair."""

    digest = hashlib.sha256(
        f"{source.strip().casefold()}\x00{page.strip().casefold()}".encode("utf-8")
    ).hexdigest()
    return f"ref-{digest[:12]}"


def parse_retrieval_block(block: object) -> EvidenceItem | None:
    """Parse one retrieval block, or return ``None`` when the header is invalid."""

    if not isinstance(block, str):
        return None
    match = _BLOCK.match(block.strip())
    if match is None:
        return None
    source = match.group("source").strip()
    page = match.group("page").strip()
    content = match.group("content").strip()
    if not source or not page or not content or source == "?" or page == "?":
        return None
    return EvidenceItem(
        reference=EvidenceReference(
            reference_id=reference_id_for(source, page),
            source=source,
            page=page,
        ),
        content=content,
    )


# Function words removed before measuring subject coverage (Spanish/English).
_STOPWORDS = frozenset(
    {
        "al", "ante", "and", "as", "at", "by", "con", "como", "cual", "desde",
        "donde", "durante", "en", "entre", "esa", "ese", "esta", "este", "estas",
        "estos", "for", "from", "hacia", "hasta", "in", "into", "its", "lo",
        "mas", "mediante", "muy", "on", "or", "para", "que", "se", "segun",
        "sin", "sobre", "su", "sus", "that", "their", "this", "to", "tras",
        "with", "about",
        # Rendering style or quality modifiers: they describe the picture, not
        # the documented subject, so they never need corpus evidence.
        "acuarela", "artistica", "artistico", "bonita", "bonito", "color",
        "colorida", "colorido", "detallada", "detallado", "estilo", "hd",
        "oleo", "realista", "realistic", "detailed", "colorful", "style",
        "fotorrealista", "photorealistic", "4k", "8k",
    }
)
_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)


def _stem(token: str) -> str:
    """Light, predictable Spanish/English number and gender folding.

    ``ibero``/``ibera``/``iberos`` -> ``iber``; ``murallas`` -> ``murall``.
    Deliberately not a full stemmer: equality of these stems is the only
    matching rule, so the behaviour stays explainable as the corpus grows.
    """

    if len(token) > 4 and token.endswith("es"):
        token = token[:-2]
    elif len(token) > 3 and token.endswith("s"):
        token = token[:-1]
    if len(token) > 4 and token[-1] in "aeo":
        token = token[:-1]
    return token


def _stems(text: str) -> set[str]:
    return {_stem(token) for token in _TOKEN.findall(_fold(text))}


def subject_terms(query: str) -> tuple[str, ...]:
    """Stemmed content words naming what the image request is about.

    Request framing ("genera una imagen de"), image verbs/nouns, function
    words and tokens shorter than three characters (unless numeric) are
    excluded.  Order is preserved and duplicates removed.
    """

    subject = retrieval_query_for_image(query)
    terms: list[str] = []
    for token in _TOKEN.findall(_fold(subject)):
        if (
            token in _IMAGE_VERBS
            or token in _IMAGE_NOUNS
            or token in _FRAMING_WORDS
            or token in _STOPWORDS
            or (len(token) < 3 and not token.isdigit())
        ):
            continue
        stem = _stem(token)
        if stem not in terms:
            terms.append(stem)
    return tuple(terms)


def covers_subject(content: str, terms: tuple[str, ...], coverage: float) -> bool:
    """Whether ``content`` mentions enough of the subject ``terms``.

    At least ``ceil(coverage * len(terms))`` distinct terms must appear.  No
    terms means the subject is unknown, which never counts as covered.
    """

    if not terms:
        return False
    required = max(1, math.ceil(coverage * len(terms) - 1e-9))
    return len(set(terms) & _stems(content)) >= required


def evidence_packet_from_blocks(
    query: str,
    blocks: Iterable[object],
    *,
    minimum_references: int = 2,
    max_items: int = 24,
    max_item_chars: int = 4000,
    subject_coverage: float | None = None,
) -> EvidencePacket:
    """Build a bounded packet from retrieval blocks in their ranked order.

    Retrieval always returns its top-k chunks, even for subjects absent from
    the corpus, so counting distinct references alone cannot tell evidence
    from noise.  When ``subject_coverage`` is given:

    * only chunks that mention the requested subject (see
      :func:`covers_subject`) become evidence, and the reference minimum is
      applied to those chunks alone; and
    * every subject term must appear in at least one of those chunks,
      otherwise the packet is empty.  This rejects requests such as "Torre
      Eiffel" where a generic word ("torre") matches but the specific one
      never does.

    The rule is lexical and corpus-independent, so it keeps working as new
    documents are ingested; it deliberately favours false negatives.
    """

    if max_items <= 0 or max_item_chars <= 0:
        raise ValueError("Evidence bounds must be positive.")
    if subject_coverage is not None and not 0 < subject_coverage <= 1:
        raise ValueError("Subject coverage must be in (0, 1].")
    terms = subject_terms(query) if subject_coverage is not None else ()
    items: list[EvidenceItem] = []
    seen: set[tuple[str, str, str]] = set()
    grounded_terms: set[str] = set()
    for block in blocks:
        item = parse_retrieval_block(block)
        if item is None:
            continue
        if subject_coverage is not None:
            if not covers_subject(item.content, terms, subject_coverage):
                continue
            grounded_terms |= set(terms) & _stems(item.content)
        content = item.content[:max_item_chars]
        key = (*item.reference.source_page_key, content)
        if key in seen:
            continue
        seen.add(key)
        items.append(item.model_copy(update={"content": content}))
        if len(items) >= max_items:
            break
    if subject_coverage is not None and grounded_terms != set(terms):
        items = []
    return build_evidence_packet(query, items, minimum_references=minimum_references)
