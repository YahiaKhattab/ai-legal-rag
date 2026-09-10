"""Deterministic language detection and output-language validation."""

from __future__ import annotations


def detect_question_language(
    text: str,
    *,
    minimum_script_ratio: float = 0.6,
) -> str:
    """Detect whether a question is primarily Arabic, English, or mixed.

    The detection is deterministic and based only on alphabetic characters.
    Numbers, punctuation, whitespace, and symbols are ignored.
    """

    letters = [
        character
        for character in text
        if character.isalpha()
    ]

    if not letters:
        return "mixed"

    arabic_letters = sum(
        _is_arabic_letter(character)
        for character in letters
    )

    latin_letters = sum(
        _is_latin_letter(character)
        for character in letters
    )

    total_script_letters = arabic_letters + latin_letters

    if total_script_letters == 0:
        return "mixed"

    arabic_ratio = (
        arabic_letters / total_script_letters
    )

    latin_ratio = (
        latin_letters / total_script_letters
    )

    if arabic_ratio >= minimum_script_ratio:
        return "ar"

    if latin_ratio >= minimum_script_ratio:
        return "en"

    return "mixed"


def answer_matches_language(
    answer: str,
    language: str,
    *,
    minimum_script_ratio: float = 0.6,
) -> bool:
    """Require the requested script while allowing numbers and punctuation."""

    if language == "mixed":
        return True

    letters = [
        character
        for character in answer
        if character.isalpha()
    ]

    if not letters:
        return False

    if language == "ar":
        matching = sum(
            _is_arabic_letter(character)
            for character in letters
        )

    elif language == "en":
        matching = sum(
            _is_latin_letter(character)
            for character in letters
        )

    else:
        return False

    return (
        matching / len(letters)
        >= minimum_script_ratio
    )


def _is_arabic_letter(character: str) -> bool:
    codepoint = ord(character)

    return (
        0x0600 <= codepoint <= 0x06FF
        or 0x0750 <= codepoint <= 0x077F
        or 0x08A0 <= codepoint <= 0x08FF
        or 0xFB50 <= codepoint <= 0xFDFF
        or 0xFE70 <= codepoint <= 0xFEFF
    )


def _is_latin_letter(character: str) -> bool:
    codepoint = ord(character)

    return (
        0x0041 <= codepoint <= 0x005A
        or 0x0061 <= codepoint <= 0x007A
        or 0x00C0 <= codepoint <= 0x024F
    )