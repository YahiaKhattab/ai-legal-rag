"""Document-level orchestration and JSONL persistence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pymupdf
from pypdf import PdfReader

from legal_rag.ingestion.chunking import chunk_page
from legal_rag.ingestion.models import ExtractionMethod, OcrText, PageRecord
from legal_rag.ingestion.native import uses_right_to_left_digit_storage
from legal_rag.ingestion.normalization import normalize_text
from legal_rag.ingestion.ocr import OcrEngine, PaddleOcrEngine
from legal_rag.ingestion.quality import ExpectedLanguage, measure_text_quality, requires_ocr


@dataclass(frozen=True, slots=True)
class IngestionSummary:
    source_file: str
    processed_pages: int
    native_pages: int
    ocr_pages: int
    failed_pages: int
    chunks: int
    pages_output: Path
    chunks_output: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class IngestionPipeline:
    def __init__(
        self,
        *,
        expected_language: ExpectedLanguage = "ar",
        ocr_engine: OcrEngine | None = None,
        maximum_chunk_characters: int = 1_200,
        chunk_overlap_characters: int = 150,
    ) -> None:
        self._expected_language = expected_language
        self._ocr_engine = ocr_engine
        self._maximum_chunk_characters = maximum_chunk_characters
        self._chunk_overlap_characters = chunk_overlap_characters

    def _get_ocr_engine(self) -> OcrEngine:
        if self._ocr_engine is None:
            ocr_language = "ar" if self._expected_language == "auto" else self._expected_language
            self._ocr_engine = PaddleOcrEngine(language=ocr_language)
        return self._ocr_engine

    def _extract_page(
        self,
        page: Any,
        *,
        native_text: str,
        source_file: str,
        document_sha256: str,
        page_number: int,
    ) -> PageRecord:
        native_quality = measure_text_quality(native_text)

        if not requires_ocr(native_quality, self._expected_language):
            reverse_arabic_digits = uses_right_to_left_digit_storage(page)
            normalized = normalize_text(
                native_text,
                reverse_arabic_digit_runs=reverse_arabic_digits,
            )
            return PageRecord(
                source_file=source_file,
                document_sha256=document_sha256,
                page_number=page_number,
                extraction_method=ExtractionMethod.NATIVE,
                raw_text=native_text,
                normalized_text=normalized,
                native_quality=native_quality,
                selected_quality=measure_text_quality(normalized),
                native_rtl_digit_correction_applied=reverse_arabic_digits,
            )

        try:
            ocr_text: OcrText = self._get_ocr_engine().extract_page(page)
            normalized = normalize_text(ocr_text.text)
            selected_quality = measure_text_quality(normalized)

            if requires_ocr(selected_quality, self._expected_language):
                raise RuntimeError("OCR output failed text-quality checks")

            return PageRecord(
                source_file=source_file,
                document_sha256=document_sha256,
                page_number=page_number,
                extraction_method=ExtractionMethod.OCR,
                raw_text=ocr_text.text,
                normalized_text=normalized,
                native_quality=native_quality,
                selected_quality=selected_quality,
                ocr_mean_confidence=ocr_text.mean_confidence,
            )
        except Exception as error:
            return PageRecord(
                source_file=source_file,
                document_sha256=document_sha256,
                page_number=page_number,
                extraction_method=ExtractionMethod.FAILED,
                raw_text="",
                normalized_text="",
                native_quality=native_quality,
                selected_quality=measure_text_quality(""),
                error=f"{type(error).__name__}: {error}",
            )

    def ingest(
        self,
        pdf_path: Path,
        output_directory: Path,
        *,
        page_limit: int | None = None,
    ) -> IngestionSummary:
        pdf_path = pdf_path.resolve()
        if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
            raise ValueError(f"Not a PDF file: {pdf_path}")
        if page_limit is not None and page_limit <= 0:
            raise ValueError("page_limit must be positive")

        output_directory.mkdir(parents=True, exist_ok=True)
        pages_output = output_directory / f"{pdf_path.stem}.pages.jsonl"
        chunks_output = output_directory / f"{pdf_path.stem}.chunks.jsonl"
        pages_temporary = pages_output.with_suffix(pages_output.suffix + ".tmp")
        chunks_temporary = chunks_output.with_suffix(chunks_output.suffix + ".tmp")
        document_sha256 = _sha256(pdf_path)

        native_pages = 0
        ocr_pages = 0
        failed_pages = 0
        chunk_count = 0
        processed_pages = 0

        with (
            pdf_path.open("rb") as native_source,
            pymupdf.open(pdf_path) as document,  # type: ignore[no-untyped-call]
            pages_temporary.open("w", encoding="utf-8", newline="\n") as pages_file,
            chunks_temporary.open("w", encoding="utf-8", newline="\n") as chunks_file,
        ):
            native_document = PdfReader(native_source, strict=False)
            native_page_count = len(native_document.pages)
            if document.page_count != native_page_count:
                raise ValueError(
                    "PDF page-count mismatch: "
                    f"PyMuPDF={document.page_count}, pypdf={native_page_count}: {pdf_path}"
                )
            if document.page_count == 0:
                raise ValueError(f"PDF contains no readable pages: {pdf_path}")
            stop = document.page_count
            if page_limit is not None:
                stop = min(stop, page_limit)

            for page_index in range(stop):
                native_text = native_document.pages[page_index].extract_text() or ""
                record = self._extract_page(
                    document[page_index],
                    native_text=native_text,
                    source_file=pdf_path.name,
                    document_sha256=document_sha256,
                    page_number=page_index + 1,
                )
                processed_pages += 1
                if record.extraction_method is ExtractionMethod.NATIVE:
                    native_pages += 1
                elif record.extraction_method is ExtractionMethod.OCR:
                    ocr_pages += 1
                else:
                    failed_pages += 1

                pages_file.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
                pages_file.flush()

                chunks = chunk_page(
                    record,
                    maximum_characters=self._maximum_chunk_characters,
                    overlap_characters=self._chunk_overlap_characters,
                )
                for chunk in chunks:
                    chunks_file.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
                chunks_file.flush()
                chunk_count += len(chunks)
                print(
                    f"{pdf_path.name}: page {page_index + 1}/{stop} "
                    f"[{record.extraction_method.value}]"
                )

        pages_temporary.replace(pages_output)
        chunks_temporary.replace(chunks_output)

        return IngestionSummary(
            source_file=pdf_path.name,
            processed_pages=processed_pages,
            native_pages=native_pages,
            ocr_pages=ocr_pages,
            failed_pages=failed_pages,
            chunks=chunk_count,
            pages_output=pages_output,
            chunks_output=chunks_output,
        )
