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


def evidence_packet_from_blocks(
    query: str,
    blocks: Iterable[object],
    *,
    minimum_references: int = 2,
    max_items: int = 24,
    max_item_chars: int = 4000,
) -> EvidencePacket:
    """Build a bounded packet from retrieval blocks in their ranked order."""

    if max_items <= 0 or max_item_chars <= 0:
        raise ValueError("Evidence bounds must be positive.")
    items: list[EvidenceItem] = []
    seen: set[tuple[str, str, str]] = set()
    for block in blocks:
        item = parse_retrieval_block(block)
        if item is None:
            continue
        content = item.content[:max_item_chars]
        key = (*item.reference.source_page_key, content)
        if key in seen:
            continue
        seen.add(key)
        items.append(item.model_copy(update={"content": content}))
        if len(items) >= max_items:
            break
    return build_evidence_packet(query, items, minimum_references=minimum_references)
