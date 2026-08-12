"""Conservative legal-structure detection with page-local source spans."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from legal_rag.ingestion.models import ExtractionMethod, PageRecord, SectionType
from legal_rag.ingestion.normalization import normalize_text

_ARABIC_SECTION = re.compile(r"^(?P<label>الباب|الفصل|القسم|الكتاب|الجزء)\b")
_ARABIC_ARTICLE = re.compile(r"^(?P<label>المادة|مادة)\s*(?:\([^)]{1,30}\)|\S{1,30})?")
_ENGLISH_SECTION = re.compile(r"^(?P<label>part|chapter|section)\b", re.IGNORECASE)
_ENGLISH_ARTICLE = re.compile(
    r"^article\s*(?:\([^)]{1,30}\)|[0-9A-Za-z.-]{1,30})?",
    re.IGNORECASE,
)
_CLAUSE = re.compile(
    r"^(?:\(?[0-9٠-٩]+\)?|[أ-يA-Za-z])\s*[-–—.):]\s+"  # noqa: RUF001
)


class LineKind(StrEnum):
    """Structural role assigned to a normalized source line."""

    HEADING = "heading"
    ARTICLE = "article"
    CLAUSE = "clause"
    PARAGRAPH = "paragraph"


@dataclass(frozen=True, slots=True)
class LineClassification:
    kind: LineKind
    section_type: SectionType | None = None
    title: str | None = None


@dataclass(frozen=True, slots=True)
class SectionSpan:
    """A contiguous page slice that never crosses a legal section boundary."""

    page_number: int
    start_char: int
    end_char: int
    section_type: SectionType
    section_title: str | None
    preferred_breaks: tuple[int, ...]


def classify_line(line: str) -> LineClassification:
    """Recognize high-confidence headings, articles, clauses, and paragraphs."""

    normalized = normalize_text(line).strip()
    if not normalized:
        return LineClassification(LineKind.PARAGRAPH)

    arabic_section = _ARABIC_SECTION.match(normalized)
    if arabic_section:
        section_type = {
            "الباب": SectionType.PART,
            "الكتاب": SectionType.PART,
            "الجزء": SectionType.PART,
            "الفصل": SectionType.CHAPTER,
            "القسم": SectionType.SECTION,
        }[arabic_section.group("label")]
        return LineClassification(LineKind.HEADING, section_type, normalized[:200])
    arabic_article = _ARABIC_ARTICLE.match(normalized)
    if arabic_article:
        return LineClassification(
            LineKind.ARTICLE,
            SectionType.ARTICLE,
            arabic_article.group(0).strip()[:200],
        )

    english_section = _ENGLISH_SECTION.match(normalized)
    if english_section:
        section_type = {
            "part": SectionType.PART,
            "chapter": SectionType.CHAPTER,
            "section": SectionType.SECTION,
        }[english_section.group("label").lower()]
        return LineClassification(LineKind.HEADING, section_type, normalized[:200])
    english_article = _ENGLISH_ARTICLE.match(normalized)
    if english_article:
        return LineClassification(
            LineKind.ARTICLE,
            SectionType.ARTICLE,
            english_article.group(0).strip()[:200],
        )
    if _CLAUSE.match(normalized):
        return LineClassification(LineKind.CLAUSE)
    return LineClassification(LineKind.PARAGRAPH)


class LegalStructureDetector:
    """Carry the latest legal heading across page boundaries."""

    def __init__(self) -> None:
        self._section_type = SectionType.DOCUMENT
        self._section_title: str | None = None

    def detect_page(self, page: PageRecord) -> list[SectionSpan]:
        if page.extraction_method is ExtractionMethod.FAILED or not page.original_text.strip():
            return []

        spans: list[SectionSpan] = []
        span_start = 0
        preferred_breaks: list[int] = []
        offset = 0

        for line in page.original_text.splitlines(keepends=True):
            line_start = offset
            offset += len(line)
            classification = classify_line(line)

            if classification.section_type is not None:
                if page.original_text[span_start:line_start].strip():
                    spans.append(
                        SectionSpan(
                            page_number=page.page_number,
                            start_char=span_start,
                            end_char=line_start,
                            section_type=self._section_type,
                            section_title=self._section_title,
                            preferred_breaks=tuple(preferred_breaks),
                        )
                    )
                self._section_type = classification.section_type
                self._section_title = classification.title
                span_start = line_start
                preferred_breaks = []
            elif classification.kind is LineKind.CLAUSE:
                preferred_breaks.append(line_start)
            elif not line.strip():
                preferred_breaks.append(offset)

        if offset < len(page.original_text):
            offset = len(page.original_text)
        if page.original_text[span_start:offset].strip():
            spans.append(
                SectionSpan(
                    page_number=page.page_number,
                    start_char=span_start,
                    end_char=offset,
                    section_type=self._section_type,
                    section_title=self._section_title,
                    preferred_breaks=tuple(preferred_breaks),
                )
            )
        return spans
