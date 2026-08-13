# ruff: noqa: RUF001 - Arabic fixtures intentionally exercise real legal text

from legal_rag.ingestion.models import ExtractionMethod, PageRecord, SectionType, TextQuality
from legal_rag.ingestion.structure import LegalStructureDetector, LineKind, classify_line

_QUALITY = TextQuality(200, 0.8, 0.0, 0.0, 0.0)


def _page(text: str, page_number: int) -> PageRecord:
    return PageRecord(
        source_file="law.pdf",
        document_id="a" * 64,
        document_version=1,
        file_hash="a" * 64,
        page_number=page_number,
        extraction_method=ExtractionMethod.NATIVE,
        native_text=text,
        original_text=text,
        normalized_text=text,
        native_quality=_QUALITY,
        selected_quality=_QUALITY,
        language="ar",
    )


def test_classifies_arabic_and_english_legal_lines() -> None:
    assert classify_line("الباب الأول").section_type is SectionType.PART
    assert classify_line("الفصل الثاني").section_type is SectionType.CHAPTER
    assert classify_line("القسم الثالث").section_type is SectionType.SECTION
    assert classify_line("المادة (١٢)").section_type is SectionType.ARTICLE
    assert classify_line("Article 4").section_type is SectionType.ARTICLE
    assert classify_line("1- first clause").kind is LineKind.CLAUSE
    assert classify_line("فقرة عادية").kind is LineKind.PARAGRAPH


def test_carries_article_metadata_to_the_following_page() -> None:
    detector = LegalStructureDetector()

    first = detector.detect_page(_page("المادة (١)\nبداية النص", 1))
    second = detector.detect_page(_page("تكملة نص المادة في الصفحة التالية", 2))

    assert first[0].section_title == "المادة (١)"
    assert second[0].section_type is SectionType.ARTICLE
    assert second[0].section_title == "المادة (١)"
