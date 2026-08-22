from legal_rag.query.answer_validator import extract_numbers, validate_numeric_claims


def test_extract_numbers_normalizes_arabic_digits() -> None:
    assert extract_numbers("المادة ٣ والمبلغ ١٢٫٥") == {"3", "12.5"}  # noqa: RUF001


def test_numeric_validator_rejects_number_absent_from_query_and_evidence() -> None:
    valid, unsupported, _ = validate_numeric_claims(
        "ما الحكم؟",
        "الحد هو 7 ملايين",
        "الحد هو خمسة ملايين",
    )

    assert valid is False
    assert "7" in unsupported or "7 million" in unsupported
