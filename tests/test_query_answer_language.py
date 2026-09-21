from legal_rag.query.answer_language import answer_matches_language


def test_arabic_answer_rejects_cjk_language_drift() -> None:
    drifted = "الم公司违反了第3条并使用了客户数据"

    assert answer_matches_language(drifted, "ar") is False


def test_arabic_answer_accepts_arabic_with_numbers() -> None:
    assert answer_matches_language("تخالف الشركة أحكام المادة 3.", "ar") is True


def test_english_answer_accepts_latin_script() -> None:
    assert answer_matches_language("The company violates article 3.", "en") is True


def test_mixed_language_does_not_enforce_one_script() -> None:
    assert answer_matches_language("المادة Article 3", "mixed") is True
