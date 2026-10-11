"""One text normalisation for training corpora and for Whisper's transcripts on the Jetson.

MELD writes typographic punctuation (he’s, …, —) while Whisper writes ASCII (he's, ...,
-). The tokenizer treats ’ and ' as different symbols, so without this the text tower
would train on one spelling and be deployed on the other.
"""

from __future__ import annotations

import re
import unicodedata

_MAP = str.maketrans({"’": "'", "‘": "'", "‛": "'", "“": '"', "”": '"', "„": '"', "–": "-", "—": "-",
                      "…": "...", " ": " "})


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "").translate(_MAP)
    return re.sub(r"\s+", " ", text).strip()
