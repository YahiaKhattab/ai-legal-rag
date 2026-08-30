"""Unit tests for legal_rag.ingestion.quality.

Covers text-quality measurement, language detection, and the OCR-routing
decision (requires_ocr), which is the function that decides whether a
page falls back to PaddleOCR.
"""

from __future__ import annotations

import pytest

from legal_rag.ingestion.models import TextQuality
from legal_rag.ingestion.quality import (
    detect_language,
    measure_text_quality,
    requires_ocr,
    resolve_language,
)


def test_measure_text_quality_pure_arabic():
    quality = measure_text_quality("القانون المدني")
    assert quality.arabic_ratio == pytest.approx(1.0)
    assert quality.latin_ratio == 0.0
    assert quality.character_count == 13  # letters only, spaces are stripped


def test_measure_text_quality_pure_english():
    quality = measure_text_quality("Civil Code")
    assert quality.latin_ratio == pytest.approx(1.0)
    assert quality.arabic_ratio == 0.0


def test_measure_text_quality_counts_replacement_and_control_chars():
    text = "abc\ufffd\x01"
    quality = measure_text_quality(text)
    assert quality.replacement_ratio == pytest.approx(1 / 5)
    assert quality.control_ratio == pytest.approx(1 / 5)


def test_measure_text_quality_empty_string_does_not_divide_by_zero():
    quality = measure_text_quality("")
    assert quality.character_count == 0
    assert quality.arabic_ratio == 0.0


def _quality(arabic_ratio=0.0, latin_ratio=0.0, character_count=100):
    return TextQuality(
        character_count=character_count,
        arabic_ratio=arabic_ratio,
        latin_ratio=latin_ratio,
        replacement_ratio=0.0,
        control_ratio=0.0,
    )


def test_detect_language_returns_unknown_for_too_little_script_evidence():
    quality = _quality(arabic_ratio=0.0, latin_ratio=0.0, character_count=1)
    assert detect_language(quality) == "unknown"


def test_detect_language_returns_arabic_when_mostly_arabic():
    quality = _quality(arabic_ratio=0.95, latin_ratio=0.05)
    assert detect_language(quality) == "ar"


def test_detect_language_returns_english_when_mostly_latin():
    quality = _quality(arabic_ratio=0.05, latin_ratio=0.95)
    assert detect_language(quality) == "en"


def test_detect_language_returns_mixed_for_balanced_scripts():
    quality = _quality(arabic_ratio=0.5, latin_ratio=0.5)
    assert detect_language(quality) == "mixed"


def test_resolve_language_honors_explicit_override():
    quality = _quality(arabic_ratio=0.95)
    assert resolve_language("en", quality) == "en"


def test_resolve_language_auto_detects_when_requested():
    quality = _quality(arabic_ratio=0.95, latin_ratio=0.05)
    assert resolve_language("auto", quality) == "ar"


def test_requires_ocr_true_when_too_few_characters():
    quality = _quality(arabic_ratio=0.9, character_count=10)
    assert requires_ocr(quality, "ar", minimum_characters=50) is True


def test_requires_ocr_true_when_replacement_ratio_too_high():
    quality = TextQuality(
        character_count=100,
        arabic_ratio=0.9,
        latin_ratio=0.0,
        replacement_ratio=0.05,
        control_ratio=0.0,
    )
    assert requires_ocr(quality, "ar") is True


def test_requires_ocr_false_for_clean_arabic_text():
    quality = _quality(arabic_ratio=0.9, character_count=200)
    assert requires_ocr(quality, "ar") is False


def test_requires_ocr_true_for_arabic_expectation_but_english_text():
    quality = _quality(latin_ratio=0.9, character_count=200)
    assert requires_ocr(quality, "ar") is True


def test_requires_ocr_auto_uses_combined_script_ratio():
    quality = _quality(arabic_ratio=0.1, latin_ratio=0.05, character_count=200)
    assert requires_ocr(quality, "auto", minimum_script_ratio=0.20) is True
    quality_good = _quality(arabic_ratio=0.3, latin_ratio=0.0, character_count=200)
    assert requires_ocr(quality_good, "auto", minimum_script_ratio=0.20) is False
