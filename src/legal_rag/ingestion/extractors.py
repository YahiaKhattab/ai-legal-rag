"""Format-specific extraction adapters producing the shared source contract."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, Literal, Protocol

import pymupdf
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from pypdf import PdfReader

from legal_rag.ingestion.models import (
    DocumentMetadata,
    ExtractionMethod,
    LocatorType,
    OcrText,
    SourceFormat,
    SourceRecord,
    SourceSegment,
    TextQuality,
)
from legal_rag.ingestion.native import uses_right_to_left_digit_storage
from legal_rag.ingestion.normalization import normalize_text
from legal_rag.ingestion.ocr import OcrEngine
from legal_rag.ingestion.quality import (
    ExpectedLanguage,
    detect_language,
    measure_text_quality,
    requires_ocr,
    resolve_language,
)

OcrLanguage = Literal["ar", "en"]
OcrEngineFactory = Callable[[OcrLanguage], OcrEngine]


class DocumentExtractor(Protocol):
    """Common interface implemented by every supported source format."""

    source_format: SourceFormat

    def extract(
        self,
        path: Path,
        metadata: DocumentMetadata,
        *,
        page_limit: int | None = None,
    ) -> Iterator[SourceRecord]: ...


class PdfExtractor:
    """Extract PDF pages with native text and conditional OCR fallback."""

    source_format = SourceFormat.PDF

    def __init__(
        self,
        *,
        expected_language: ExpectedLanguage,
        ocr_engine_factory: OcrEngineFactory,
    ) -> None:
        self._expected_language = expected_language
        self._ocr_engine_factory = ocr_engine_factory
        self._auto_ocr_language: OcrLanguage | None = None

    def _ocr_candidate(
        self,
        page: Any,
        language: OcrLanguage,
    ) -> tuple[OcrText, TextQuality]:
        result = self._ocr_engine_factory(language).extract_page(page)
        quality = measure_text_quality(normalize_text(result.text))
        return result, quality

    def _automatic_ocr(
        self,
        page: Any,
        native_quality: TextQuality,
    ) -> tuple[OcrText, TextQuality]:
        native_language = detect_language(native_quality)
        preferred: OcrLanguage | None = None
        if native_language in ("ar", "en"):
            preferred = native_language
        elif self._auto_ocr_language is not None:
            preferred = self._auto_ocr_language

        if preferred is not None:
            alternatives: tuple[OcrLanguage, OcrLanguage] = (
                preferred,
                "en" if preferred == "ar" else "ar",
            )
            for language in alternatives:
                result, quality = self._ocr_candidate(page, language)
                if not requires_ocr(quality, language):
                    self._auto_ocr_language = language
                    return result, quality
            raise RuntimeError("Arabic and English OCR outputs failed text-quality checks")

        candidates: list[tuple[OcrText, TextQuality, OcrLanguage]] = []
        for language in ("ar", "en"):
            result, quality = self._ocr_candidate(page, language)
            if not requires_ocr(quality, language):
                candidates.append((result, quality, language))
        if not candidates:
            raise RuntimeError("Arabic and English OCR outputs failed text-quality checks")

        def candidate_score(
            candidate: tuple[OcrText, TextQuality, OcrLanguage],
        ) -> tuple[float, float, int]:
            result, quality, language = candidate
            script_ratio = quality.arabic_ratio if language == "ar" else quality.latin_ratio
            return script_ratio, result.mean_confidence or 0.0, quality.character_count

        result, quality, language = max(candidates, key=candidate_score)
        self._auto_ocr_language = language
        return result, quality

    def extract_page(
        self,
        page: Any,
        *,
        native_text: str,
        metadata: DocumentMetadata,
        page_number: int,
    ) -> SourceRecord:
        native_quality = measure_text_quality(native_text)
        if not requires_ocr(native_quality, self._expected_language):
            reverse_digits = uses_right_to_left_digit_storage(page)
            normalized = normalize_text(
                native_text,
                reverse_arabic_digit_runs=reverse_digits,
            )
            selected_quality = measure_text_quality(normalized)
            return SourceRecord(
                source_file=metadata.source_file,
                document_id=metadata.document_id,
                document_version=metadata.document_version,
                file_hash=metadata.file_hash,
                extraction_method=ExtractionMethod.NATIVE,
                native_text=native_text,
                original_text=native_text,
                normalized_text=normalized,
                native_quality=native_quality,
                selected_quality=selected_quality,
                language=resolve_language(self._expected_language, selected_quality),
                source_format=SourceFormat.PDF,
                locator_type=LocatorType.PAGE,
                locator_start=page_number,
                locator_end=page_number,
                page_number=page_number,
                native_rtl_digit_correction_applied=reverse_digits,
            )

        try:
            if self._expected_language == "auto":
                ocr_text, selected_quality = self._automatic_ocr(page, native_quality)
            else:
                ocr_text, selected_quality = self._ocr_candidate(
                    page,
                    self._expected_language,
                )
            normalized = normalize_text(ocr_text.text)
            if requires_ocr(selected_quality, self._expected_language):
                raise RuntimeError("OCR output failed text-quality checks")
            return SourceRecord(
                source_file=metadata.source_file,
                document_id=metadata.document_id,
                document_version=metadata.document_version,
                file_hash=metadata.file_hash,
                extraction_method=ExtractionMethod.OCR,
                native_text=native_text,
                original_text=ocr_text.text,
                normalized_text=normalized,
                native_quality=native_quality,
                selected_quality=selected_quality,
                language=resolve_language(self._expected_language, selected_quality),
                source_format=SourceFormat.PDF,
                locator_type=LocatorType.PAGE,
                locator_start=page_number,
                locator_end=page_number,
                page_number=page_number,
                ocr_mean_confidence=ocr_text.mean_confidence,
            )
        except Exception as error:
            return SourceRecord(
                source_file=metadata.source_file,
                document_id=metadata.document_id,
                document_version=metadata.document_version,
                file_hash=metadata.file_hash,
                extraction_method=ExtractionMethod.FAILED,
                native_text=native_text,
                original_text="",
                normalized_text="",
                native_quality=native_quality,
                selected_quality=measure_text_quality(""),
                language=resolve_language(self._expected_language, native_quality),
                source_format=SourceFormat.PDF,
                locator_type=LocatorType.PAGE,
                locator_start=page_number,
                locator_end=page_number,
                page_number=page_number,
                error=f"{type(error).__name__}: {error}",
            )

    def extract(
        self,
        path: Path,
        metadata: DocumentMetadata,
        *,
        page_limit: int | None = None,
    ) -> Iterator[SourceRecord]:
        with (
            path.open("rb") as native_source,
            pymupdf.open(path) as document,  # type: ignore[no-untyped-call]
        ):
            native_document = PdfReader(native_source, strict=False)
            native_page_count = len(native_document.pages)
            if document.page_count != native_page_count:
                raise ValueError(
                    "PDF page-count mismatch: "
                    f"PyMuPDF={document.page_count}, pypdf={native_page_count}: {path}"
                )
            stop = (
                document.page_count if page_limit is None else min(document.page_count, page_limit)
            )
            for page_index in range(stop):
                try:
                    native_text = native_document.pages[page_index].extract_text() or ""
                except Exception:
                    native_text = ""
                yield self.extract_page(
                    document[page_index],
                    native_text=native_text,
                    metadata=metadata,
                    page_number=page_index + 1,
                )


def _continuous_record(
    *,
    metadata: DocumentMetadata,
    text: str,
    method: ExtractionMethod,
    locator_type: LocatorType,
    segments: tuple[SourceSegment, ...],
    expected_language: ExpectedLanguage,
) -> SourceRecord:
    native_quality = measure_text_quality(text)
    normalized = normalize_text(text)
    selected_quality = measure_text_quality(normalized)
    return SourceRecord(
        source_file=metadata.source_file,
        document_id=metadata.document_id,
        document_version=metadata.document_version,
        file_hash=metadata.file_hash,
        extraction_method=method,
        native_text=text,
        original_text=text,
        normalized_text=normalized,
        native_quality=native_quality,
        selected_quality=selected_quality,
        language=resolve_language(expected_language, selected_quality),
        source_format=metadata.source_format,
        locator_type=locator_type,
        locator_start=segments[0].locator_start,
        locator_end=segments[-1].locator_end,
        source_segments=segments,
    )


class TxtExtractor:
    """Extract one UTF-8 text document while preserving line locators."""

    source_format = SourceFormat.TXT

    def __init__(self, *, expected_language: ExpectedLanguage) -> None:
        self._expected_language = expected_language

    def extract(
        self,
        path: Path,
        metadata: DocumentMetadata,
        *,
        page_limit: int | None = None,
    ) -> Iterator[SourceRecord]:
        if page_limit is not None:
            raise ValueError("--pages is supported only for PDF input")
        text = path.read_bytes().decode("utf-8-sig")
        segments: list[SourceSegment] = []
        offset = 0
        for line_number, line in enumerate(text.splitlines(keepends=True), start=1):
            end = offset + len(line)
            segments.append(SourceSegment(offset, end, line_number, line_number, "line"))
            offset = end
        if offset < len(text) or not segments:
            line_number = len(segments) + 1
            segments.append(SourceSegment(offset, len(text), line_number, line_number, "line"))
        yield _continuous_record(
            metadata=metadata,
            text=text,
            method=ExtractionMethod.TXT,
            locator_type=LocatorType.LINE,
            segments=tuple(segments),
            expected_language=self._expected_language,
        )


def _docx_blocks(path: Path) -> tuple[str, tuple[SourceSegment, ...]]:
    document = Document(str(path))
    blocks: list[tuple[str, str]] = []
    for item in document.iter_inner_content():
        if isinstance(item, Paragraph):
            blocks.append((item.text, "paragraph"))
        elif isinstance(item, Table):
            for row in item.rows:
                blocks.append(("\t".join(cell.text for cell in row.cells), "table_row"))

    parts: list[str] = []
    segments: list[SourceSegment] = []
    offset = 0
    for block_number, (block_text, kind) in enumerate(blocks, start=1):
        rendered = block_text + "\n"
        parts.append(rendered)
        end = offset + len(rendered)
        segments.append(SourceSegment(offset, end, block_number, block_number, kind))
        offset = end
    text = "".join(parts)
    if not text.strip():
        raise ValueError(f"DOCX contains no extractable body text: {path}")
    return text, tuple(segments)


class DocxExtractor:
    """Extract body paragraphs and table rows in Word document order."""

    source_format = SourceFormat.DOCX

    def __init__(self, *, expected_language: ExpectedLanguage) -> None:
        self._expected_language = expected_language

    def extract(
        self,
        path: Path,
        metadata: DocumentMetadata,
        *,
        page_limit: int | None = None,
    ) -> Iterator[SourceRecord]:
        if page_limit is not None:
            raise ValueError("--pages is supported only for PDF input")
        text, segments = _docx_blocks(path)
        yield _continuous_record(
            metadata=metadata,
            text=text,
            method=ExtractionMethod.DOCX,
            locator_type=LocatorType.BLOCK,
            segments=segments,
            expected_language=self._expected_language,
        )
