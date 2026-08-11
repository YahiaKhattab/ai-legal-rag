from legal_rag.ingestion.quality import measure_text_quality, requires_ocr


def test_corrupted_cbe_text_requires_ocr() -> None:
    text = "UNOM ¥ sL W‡M‡‡ ±¥¥≤ WM Âd;« ≤∑ v —œUB"

    assert requires_ocr(measure_text_quality(text), "ar")


def test_arabic_native_text_is_accepted() -> None:
    phrase = "قانون تنظيم وتنمية استخدام التكنولوجيا المالية "
    text = phrase * 8

    assert not requires_ocr(measure_text_quality(text), "ar")


def test_short_page_requires_ocr() -> None:
    assert requires_ocr(measure_text_quality("صفحة قصيرة"), "ar")
