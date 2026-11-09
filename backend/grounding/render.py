"""Deterministic image-engine prompts derived from the canonical instruction.

Two prompt formats exist:

* ``canonical_json``: the engine receives the full canonical instruction. Suited
  to language-model image engines (Gemini) that read long structured text.
* ``diffusion``: the engine receives a short English prompt for CLIP-based
  diffusion models (Stable Diffusion via ComfyUI/A1111), which read only about
  77 tokens and mostly English.  With the JSON they drew anachronisms (castle
  battlements) that the instruction itself forbids.

The diffusion prompt is a pure function of the canonical JSON: it uses only the
documented facts (their ``text_en`` rendering, else the original text) plus
style and view.  Inferred elements, uncertainties and forbidden additions never
enter it, so nothing outside the evidence-backed facts can reach the engine.
Provenance re-derives it to prove the enforced prompt matches the hash.
"""

from __future__ import annotations

import json
import re


PROMPT_FORMAT_CANONICAL = "canonical_json"
PROMPT_FORMAT_DIFFUSION = "diffusion"
PROMPT_FORMATS = frozenset({PROMPT_FORMAT_CANONICAL, PROMPT_FORMAT_DIFFUSION})

DIFFUSION_PREFIX = "archaeological interpretive reconstruction"
DIFFUSION_SUFFIX = "realistic, historically accurate"
# CLIP reads ~77 tokens; ~55 words leaves room for prefix, suffix and commas.
MAX_DIFFUSION_WORDS = 55

_WHITESPACE = re.compile(r"\s+")
# Figure captions such as "Figura 12.- " carry no visual content.
_FIGURE_PREFIX = re.compile(r"^(fig(ura|ure)?\.?\s*\d+[a-z]?\s*[.:-]+\s*)", re.IGNORECASE)


class InvalidRenderPrompt(ValueError):
    """The canonical instruction cannot produce an engine prompt."""


def _clean(text: object) -> str:
    if not isinstance(text, str):
        return ""
    text = _WHITESPACE.sub(" ", text).strip().rstrip(".")
    return _FIGURE_PREFIX.sub("", text).strip()


def render_diffusion_prompt(canonical_json: str, *, max_words: int = MAX_DIFFUSION_WORDS) -> str:
    try:
        canonical = json.loads(canonical_json)
    except (TypeError, json.JSONDecodeError):
        raise InvalidRenderPrompt("Canonical instruction is not JSON.") from None
    if not isinstance(canonical, dict):
        raise InvalidRenderPrompt("Canonical instruction must be an object.")
    facts = canonical.get("documented_facts")
    if not isinstance(facts, list) or not facts:
        raise InvalidRenderPrompt("Canonical instruction has no documented facts.")

    head = [DIFFUSION_PREFIX]
    for key in ("style", "view"):
        value = _clean(canonical.get(key))
        if value:
            head.append(value)

    budget = max_words - sum(len(part.split()) for part in head) - len(DIFFUSION_SUFFIX.split())
    body: list[str] = []
    seen: set[str] = set()
    for fact in facts:
        if not isinstance(fact, dict):
            continue
        text = _clean(fact.get("text_en")) or _clean(fact.get("text"))
        if not text or text.casefold() in seen:
            continue
        words = text.split()
        if len(words) > budget:
            if not body:  # Always keep at least the start of the first fact.
                body.append(" ".join(words[: max(budget, 1)]))
            break
        body.append(text)
        seen.add(text.casefold())
        budget -= len(words)
    if not body:
        raise InvalidRenderPrompt("Documented facts produced an empty prompt.")
    return ", ".join([*head, *body, DIFFUSION_SUFFIX])


def render_engine_prompt(canonical_json: str, prompt_format: str) -> str | None:
    """Return the engine prompt for ``prompt_format``; ``None`` means canonical JSON."""

    if prompt_format == PROMPT_FORMAT_CANONICAL:
        return None
    if prompt_format == PROMPT_FORMAT_DIFFUSION:
        return render_diffusion_prompt(canonical_json)
    raise InvalidRenderPrompt("Unsupported prompt format.")
