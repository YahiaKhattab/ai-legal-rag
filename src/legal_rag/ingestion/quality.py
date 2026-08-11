"""Evidence-based text quality measurements and OCR routing."""

from __future__ import annotations

import unicodedata
from typing import Literal

from legal_rag.ingestion.models import TextQuality

ExpectedLanguage = Literal["ar", "en", "auto"]

_ARABIC_RANGES = (
    ("\u0600", "\u06ff"),
    ("\u0750", "\u077f"),
    ("\u08a0", "\u08ff"),
    ("\ufb50", "\ufdff"),
    ("\ufe70", "\ufeff"),
)


def _is_arabic(character: str) -> bool:
    return any(start <= character <= end for start, end in _ARABIC_RANGES)


def measure_text_quality(text: str) -> TextQuality:
    compact = "".join(character for character in text if not character.isspace())
    count = len(compact)
    denominator = max(count, 1)
    arabic_count = sum(_is_arabic(character) for character in compact)
    latin_count = sum(
        "LATIN" in unicodedata.name(character, "") and character.isalpha() for character in compact
    )
    replacement_count = compact.count("\ufffd")
    control_count = sum(unicodedata.category(character) == "Cc" for character in compact)

    return TextQuality(
        character_count=count,
        arabic_ratio=arabic_count / denominator,
        latin_ratio=latin_count / denominator,
        replacement_ratio=replacement_count / denominator,
        control_ratio=control_count / denominator,
    )


def requires_ocr(
    quality: TextQuality,
    expected_language: ExpectedLanguage = "ar",
    *,
    minimum_characters: int = 50,
    minimum_script_ratio: float = 0.20,
) -> bool:
    if quality.character_count < minimum_characters:
        return True
    if quality.replacement_ratio > 0.01 or quality.control_ratio > 0.01:
        return True

    if expected_language == "ar":
        return quality.arabic_ratio < minimum_script_ratio
    if expected_language == "en":
        return quality.latin_ratio < minimum_script_ratio
    return max(quality.arabic_ratio, quality.latin_ratio) < minimum_script_ratio
