"""Unit tests for legal_rag.query.answer_validator.

This module protects legal numeric thresholds from being silently altered
by the LLM, so it is the highest-value place in the whole `query/`
package to have thorough unit tests -- a bug here means the system could
present a wrong legal number as fact.
"""

from __future__ import annotations

from legal_rag.query.answer_validator import (
    extract_arabic_number_words,
    extract_numbers,
    validate_numeric_claims,
)


# ------------------------------------------------------------- extract_numbers


def test_extract_numbers_finds_western_digits():
    assert extract_numbers("The fee is 500 EGP") == {"500"}


def test_extract_numbers_finds_arabic_indic_digits_and_normalizes_them():
    assert extract_numbers("الغرامة ٥٠٠ جنيه") == {"500"}


def test_extract_numbers_strips_thousands_separators():
    assert extract_numbers("1,000,000") == {"1000000"}


def test_extract_numbers_handles_multiple_numbers():
    assert extract_numbers("3 and 7 and 12") == {"3", "7", "12"}


def test_extract_numbers_empty_text_returns_empty_set():
    assert extract_numbers("") == set()


# ------------------------------------------------------ extract_arabic_number_words


def test_extract_arabic_number_words_recognizes_known_words():
    assert extract_arabic_number_words("خمسة ملايين جنيه") == {5}


def test_extract_arabic_number_words_ignores_unknown_words():
    assert extract_arabic_number_words("كتاب القانون المدني") == set()


# --------------------------------------------------------- validate_numeric_claims


def test_valid_when_answer_reuses_a_user_supplied_amount():
    query = "قضيتي قيمتها 3 مليون جنيه، هل تختص بها المحكمة الجزئية؟"
    evidence = "الدعاوى التي تقل قيمتها عن خمسة ملايين جنيه تختص بها المحكمة الجزئية."
    answer = "قضيتك البالغة 3 ملايين تقع ضمن اختصاص الدائرة الابتدائية."

    is_valid, unsupported, _ = validate_numeric_claims(query, answer, evidence)

    assert is_valid is True
    assert unsupported == set()


def test_invalid_when_answer_replaces_evidence_threshold_with_query_amount():
    query = "قضيتي قيمتها 3 مليون جنيه"
    evidence = "الدعاوى التي تقل قيمتها عن خمسة ملايين جنيه تختص بها المحكمة الجزئية."
    answer = "الدعاوى التي تقل قيمتها عن 3 ملايين تختص بها المحكمة الجزئية."

    is_valid, unsupported, _ = validate_numeric_claims(query, answer, evidence)

    assert is_valid is False
    assert "3 million" in unsupported


def test_invalid_when_answer_invents_a_number_not_in_query_or_evidence():
    query = "What is the appeal period?"
    evidence = "The appeal period is thirty days."
    answer = "The appeal period is 45 days."

    is_valid, unsupported, _ = validate_numeric_claims(query, answer, evidence)

    assert is_valid is False
    assert "45" in unsupported


def test_valid_when_answer_number_comes_directly_from_evidence():
    query = "What is the fee?"
    evidence = "The filing fee is 500 EGP."
    answer = "The filing fee is 500 EGP."

    is_valid, unsupported, _ = validate_numeric_claims(query, answer, evidence)

    assert is_valid is True


def test_valid_when_no_numbers_present_anywhere():
    is_valid, unsupported, _ = validate_numeric_claims(
        "What does article 5 say?", "It defines general obligations.", "Article 5 text."
    )
    assert is_valid is True
    assert unsupported == set()


def test_evidence_threshold_word_form_is_recognized_english():
    query = "case value"
    evidence = "Claims under 5 million EGP fall under the first-instance court."
    answer = "Claims under 5 million EGP fall under the first-instance court."

    is_valid, unsupported, evidence_numbers = validate_numeric_claims(query, answer, evidence)

    assert is_valid is True
    assert "5 million" in evidence_numbers
