"""Conservative retrieval normalization while raw source text is preserved."""

from __future__ import annotations

import re
import unicodedata

_ARABIC_DIACRITICS = re.compile(
    "[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]"
)
_ARABIC_DIGIT_RUN = re.compile("[\u0660-\u0669\u06f0-\u06f9]+")
_PRIVATE_USE_CHARACTERS = re.compile(
    "[\ue000-\uf8ff\U000f0000-\U000ffffd\U00100000-\U0010fffd]+"
)
_FORMATTING_CHARACTERS = (
    "\u0640\u200b\u200c\u200d\u200e\u200f\u202a\u202b\u202c\u202d\u202e"
    "\u2066\u2067\u2068\u2069"
)
_REMOVED_FORMATTING = dict.fromkeys(map(ord, _FORMATTING_CHARACTERS), None)
_ARABIC_EQUIVALENTS = str.maketrans(
    {
        "أ": "\u0627",
        "إ": "\u0627",
        "آ": "\u0627",
        "ٱ": "\u0627",
        "ى": "ي",
        "ی": "ي",
        "ک": "ك",
    }
)


def normalize_text(text: str, *, reverse_arabic_digit_runs: bool = False) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.translate(_REMOVED_FORMATTING)
    normalized = _PRIVATE_USE_CHARACTERS.sub(" ", normalized)
    normalized = _ARABIC_DIACRITICS.sub("", normalized)
    normalized = normalized.translate(_ARABIC_EQUIVALENTS)
    if reverse_arabic_digit_runs:
        normalized = _ARABIC_DIGIT_RUN.sub(
            lambda match: match.group(0)[::-1],
            normalized,
        )

    lines = [" ".join(line.split()) for line in normalized.splitlines()]
    nonempty_lines = [line for line in lines if line]
    return "\n".join(nonempty_lines).strip()
