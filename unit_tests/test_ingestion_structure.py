"""Unit tests for legal_rag.ingestion.structure (legal heading/article
detection and LegalStructureDetector).
"""

from __future__ import annotations

from legal_rag.ingestion.models import (
    ExtractionMethod,
    LocatorType,
    SectionType,
    SourceFormat,
    TextQuality,
)
from legal_rag.ingestion.structure import LegalStructureDetector, LineKind, classify_line

_QUALITY = TextQuality(
    character_count=10, arabic_ratio=1.0, latin_ratio=0.0, replacement_ratio=0.0, control_ratio=0.0
)


def _source_record(text: str, *, page_number: int | None = 1):
    from legal_rag.ingestion.models import SourceRecord

    return SourceRecord(
        source_file="doc.pdf",
        document_id="doc-1",
        document_version=1,
        file_hash="hash",
        extraction_method=ExtractionMethod.NATIVE,
        native_text=text,
        original_text=text,
        normalized_text=text,
        native_quality=_QUALITY,
        selected_quality=_QUALITY,
        language="ar",
        source_format=SourceFormat.PDF,
        locator_type=LocatorType.PAGE,
        locator_start=1,
        locator_end=1,
        page_number=page_number,
    )


# ------------------------------------------------------------- classify_line


def test_classify_line_recognizes_arabic_chapter_heading():
    result = classify_line("الفصل الأول: أحكام عامة")
    assert result.kind == LineKind.HEADING
    assert result.section_type is SectionType.CHAPTER


def test_classify_line_recognizes_arabic_part_heading():
    result = classify_line("الباب الثاني")
    assert result.section_type is SectionType.PART


def test_classify_line_recognizes_arabic_article():
    result = classify_line("المادة (5): يجب على كل شخص...")
    assert result.kind == LineKind.ARTICLE
    assert result.section_type is SectionType.ARTICLE


def test_classify_line_recognizes_english_section_heading():
    result = classify_line("Chapter 3: General Provisions")
    assert result.kind == LineKind.HEADING
    assert result.section_type is SectionType.CHAPTER


def test_classify_line_recognizes_english_article():
    result = classify_line("Article 12 - Definitions")
    assert result.kind == LineKind.ARTICLE
    assert result.section_type is SectionType.ARTICLE


def test_classify_line_recognizes_numbered_clause():
    result = classify_line("1- This is a clause")
    assert result.kind == LineKind.CLAUSE
    assert result.section_type is None


def test_classify_line_falls_back_to_paragraph():
    result = classify_line("This is a normal sentence with no structure marker.")
    assert result.kind == LineKind.PARAGRAPH


def test_classify_line_blank_line_is_paragraph():
    result = classify_line("   ")
    assert result.kind == LineKind.PARAGRAPH
    assert result.title is None


# ------------------------------------------------------ LegalStructureDetector


def test_detect_source_returns_empty_for_failed_extraction():
    import dataclasses

    record = _source_record("Article 1: text")
    failed = dataclasses.replace(record, extraction_method=ExtractionMethod.FAILED)
    detector = LegalStructureDetector()
    assert detector.detect_source(failed) == []


def test_detect_source_returns_empty_for_blank_text():
    detector = LegalStructureDetector()
    assert detector.detect_source(_source_record("   \n  ")) == []


def test_detect_source_creates_one_span_for_flat_text():
    detector = LegalStructureDetector()
    text = "Just some plain paragraph text with no legal headings at all."
    spans = detector.detect_source(_source_record(text))

    assert len(spans) == 1
    assert spans[0].section_type is SectionType.DOCUMENT
    assert spans[0].start_char == 0
    assert spans[0].end_char == len(text)


def test_detect_source_splits_on_heading_boundaries():
    text = "Chapter 1: Intro\nSome text under chapter 1.\nChapter 2: More\nMore text."
    detector = LegalStructureDetector()
    spans = detector.detect_source(_source_record(text))

    # First span is the DOCUMENT-level text preceding "Chapter 1" heading line
    # itself becomes the start of a new heading-tagged span; then a second
    # span is created once "Chapter 2" appears.
    section_types = [span.section_type for span in spans]
    assert SectionType.CHAPTER in section_types


def test_detector_carries_latest_heading_across_calls_pages():
    """A heading detected on page 1 should still apply to page 2 text."""
    detector = LegalStructureDetector()
    page1 = _source_record("Chapter 5: Carried Over\n", page_number=1)
    page2 = _source_record("Continuing text with no new heading.\n", page_number=2)

    detector.detect_source(page1)
    spans_page2 = detector.detect_source(page2)

    assert spans_page2[0].section_type is SectionType.CHAPTER
    assert spans_page2[0].section_title is not None


def test_detect_page_is_a_compatibility_alias_for_detect_source():
    detector_a = LegalStructureDetector()
    detector_b = LegalStructureDetector()
    text = "Article 9: Something"
    record = _source_record(text)

    assert detector_a.detect_source(record) == detector_b.detect_page(record)
