from legal_rag.ingestion.quality import detect_language, measure_text_quality, requires_ocr


def test_corrupted_cbe_text_requires_ocr() -> None:
    text = "UNOM ¥ sL W‡M‡‡ ±¥¥≤ WM Âd;« ≤∑ v —œUB"

    assert requires_ocr(measure_text_quality(text), "ar")


def test_arabic_native_text_is_accepted() -> None:
    phrase = "قانون تنظيم وتنمية استخدام التكنولوجيا المالية "
    text = phrase * 8

    assert not requires_ocr(measure_text_quality(text), "ar")


def test_short_page_requires_ocr() -> None:
    assert requires_ocr(measure_text_quality("صفحة قصيرة"), "ar")


def test_auto_language_detects_arabic_english_mixed_and_unknown() -> None:
    assert detect_language(measure_text_quality("المادة الأولى من القانون")) == "ar"
    assert detect_language(measure_text_quality("Article one of the banking law")) == "en"
    assert detect_language(measure_text_quality("المادة الأولى Article one")) == "mixed"
    assert detect_language(measure_text_quality("194 / 2020")) == "unknown"


def test_auto_quality_accepts_balanced_bilingual_text() -> None:
    text = ("المادة القانونية Article banking regulation " * 5).strip()

    assert not requires_ocr(measure_text_quality(text), "auto")
