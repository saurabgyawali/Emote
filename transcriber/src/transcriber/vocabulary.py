"""Personal vocabulary: terms Whisper should prefer, passed to faster-whisper as `hotwords`.

Whisper already knows everyday English; what it gets wrong are names and jargon it rarely
saw ("CUDA" came out as "qdai", "go.sh" as "go pass go.sh"). faster-whisper puts the
hotwords in the decoder prompt ahead of the context prompt, and each half of that prompt is
capped at 223 tokens, so the list is kept short: MAX_CHARS is roughly 100 tokens.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)
MAX_CHARS = 400


def load_vocabulary(path: Path) -> list[str]:
    """One term per line; blank lines and # comments are ignored; duplicates dropped."""
    if not path.exists():
        return []
    terms: list[str] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        term = line.split("#", 1)[0].strip()
        if term and term.lower() not in seen:
            seen.add(term.lower())
            terms.append(term)
    return terms


def hotwords(terms: list[str]) -> str | None:
    """Comma-joined terms, earliest first, cut at MAX_CHARS so the prompt never overflows."""
    kept: list[str] = []
    length = 0
    for term in terms:
        if length + len(term) + 2 > MAX_CHARS:
            logger.warning("Vocabulary is longer than %d characters; using the first %d of %d terms.",
                           MAX_CHARS, len(kept), len(terms))
            break
        kept.append(term)
        length += len(term) + 2
    return ", ".join(kept) or None
