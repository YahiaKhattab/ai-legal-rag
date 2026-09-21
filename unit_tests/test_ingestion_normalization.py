"""Unit tests for legal_rag.ingestion.normalization.normalize_text.

normalize_text is pure string-in/string-out logic, which makes it one of
the easiest and highest-value functions to unit test thoroughly.
"""

from __future__ import annotations

from legal_rag.ingestion.normalization import normalize_text


def test_removes_arabic_diacritics():
    text = "الْقَانُونُ"
    assert normalize_text(text) == "القانون"


def test_normalizes_hamza_and_alif_variants():
    assert normalize_text("أحمد") == "احمد"
    assert normalize_text("إحسان") == "احسان"
    assert normalize_text("آدم") == "ادم"


def test_normalizes_alif_maqsura_and_farsi_yeh():
    assert normalize_text("على") == "علي"
    assert normalize_text("کیف") == "كيف"  # noqa: RUF001 (Farsi kaf/yeh -> Arabic)


def test_removes_tatweel_and_bidi_control_characters():
    text = "الحـــق\u200e\u200f"
    assert normalize_text(text) == "الحق"


def test_removes_private_use_area_characters():
    text = "abc\ue000def"
    assert normalize_text(text) == "abc def"


def test_collapses_internal_whitespace_per_line():
    text = "hello    world"
    assert normalize_text(text) == "hello world"


def test_drops_blank_lines_and_strips_result():
    text = "\n\nline one\n\n\nline two\n\n"
    assert normalize_text(text) == "line one\nline two"


def test_reverse_arabic_digit_runs_when_enabled():
    # "123" written with Arabic-Indic digits, reversed because the flag is set.
    text = "\u0661\u0662\u0663"  # ١٢٣
    result = normalize_text(text, reverse_arabic_digit_runs=True)
    assert result == "\u0663\u0662\u0661"  # ٣٢١


def test_reverse_arabic_digit_runs_disabled_by_default():
    text = "\u0661\u0662\u0663"
    result = normalize_text(text)
    assert result == text


def test_empty_and_whitespace_only_input_returns_empty_string():
    assert normalize_text("") == ""
    assert normalize_text("   \n\n  \t ") == ""
