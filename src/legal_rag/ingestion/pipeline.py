"""Document orchestration, idempotency, and atomic JSON persistence."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pymupdf
from pypdf import PdfReader

from legal_rag.ingestion.chunking import (
    DEFAULT_CHUNKING_CONFIG,
    PIPELINE_VERSION,
    chunk_page_sections,
)
from legal_rag.ingestion.models import (
    ChunkingConfig,
    DocumentMetadata,
    ExtractionMethod,
    IngestionStatus,
    OcrText,
    PageRecord,
    TextQuality,
)
from legal_rag.ingestion.native import uses_right_to_left_digit_storage
from legal_rag.ingestion.normalization import normalize_text
from legal_rag.ingestion.ocr import OcrEngine, PaddleOcrEngine
from legal_rag.ingestion.quality import ExpectedLanguage, measure_text_quality, requires_ocr
from legal_rag.ingestion.structure import LegalStructureDetector
from legal_rag.ingestion.tokenization import (
    DEFAULT_E5_REVISION,
    DEFAULT_E5_TOKENIZER,
    E5TokenCounter,
    TokenCounter,
)
from legal_rag.ingestion.validation import DEFAULT_MAXIMUM_PDF_BYTES, validate_pdf


@dataclass(frozen=True, slots=True)
class IngestionSummary:
    """Compact result returned to the CLI and callers."""

    status: IngestionStatus
    source_file: str
    document_id: str
    processed_pages: int
    native_pages: int
    ocr_pages: int
    failed_pages: int
    chunks: int
    pages_output: Path
    chunks_output: Path
    report_output: Path


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


class IngestionPipeline:
    """Coordinate validation, extraction, structure detection, and chunking."""

    def __init__(
        self,
        *,
        expected_language: ExpectedLanguage = "ar",
        ocr_engine: OcrEngine | None = None,
        token_counter: TokenCounter | None = None,
        tokenizer_identifier: str = DEFAULT_E5_TOKENIZER,
        tokenizer_revision: str = DEFAULT_E5_REVISION,
        chunking: ChunkingConfig = DEFAULT_CHUNKING_CONFIG,
        maximum_pdf_bytes: int = DEFAULT_MAXIMUM_PDF_BYTES,
    ) -> None:
        self._expected_language = expected_language
        self._ocr_engine = ocr_engine
        self._token_counter = token_counter
        self._tokenizer_identifier = tokenizer_identifier
        self._tokenizer_revision = tokenizer_revision
        self._chunking = chunking
        self._maximum_pdf_bytes = maximum_pdf_bytes

    def _get_ocr_engine(self) -> OcrEngine:
        if self._ocr_engine is None:
            ocr_language = "ar" if self._expected_language == "auto" else self._expected_language
            self._ocr_engine = PaddleOcrEngine(language=ocr_language)
        return self._ocr_engine

    def _get_token_counter(self) -> TokenCounter:
        if self._token_counter is None:
            self._token_counter = E5TokenCounter.load(
                self._tokenizer_identifier,
                revision=self._tokenizer_revision,
            )
        return self._token_counter

    def _page_language(self, quality: TextQuality) -> str:
        if self._expected_language != "auto":
            return self._expected_language
        if quality.arabic_ratio > quality.latin_ratio:
            return "ar"
        if quality.latin_ratio > quality.arabic_ratio:
            return "en"
        return "unknown"

    def _extract_page(
        self,
        page: Any,
        *,
        native_text: str,
        metadata: DocumentMetadata,
        page_number: int,
    ) -> PageRecord:
        native_quality = measure_text_quality(native_text)

        if not requires_ocr(native_quality, self._expected_language):
            reverse_arabic_digits = uses_right_to_left_digit_storage(page)
            normalized = normalize_text(
                native_text,
                reverse_arabic_digit_runs=reverse_arabic_digits,
            )
            selected_quality = measure_text_quality(normalized)
            return PageRecord(
                source_file=metadata.source_file,
                document_id=metadata.document_id,
                document_version=metadata.document_version,
                file_hash=metadata.file_hash,
                page_number=page_number,
                extraction_method=ExtractionMethod.NATIVE,
                native_text=native_text,
                original_text=native_text,
                normalized_text=normalized,
                native_quality=native_quality,
                selected_quality=selected_quality,
                language=self._page_language(selected_quality),
                native_rtl_digit_correction_applied=reverse_arabic_digits,
            )

        try:
            ocr_text: OcrText = self._get_ocr_engine().extract_page(page)
            normalized = normalize_text(ocr_text.text)
            selected_quality = measure_text_quality(normalized)

            if requires_ocr(selected_quality, self._expected_language):
                raise RuntimeError("OCR output failed text-quality checks")

            return PageRecord(
                source_file=metadata.source_file,
                document_id=metadata.document_id,
                document_version=metadata.document_version,
                file_hash=metadata.file_hash,
                page_number=page_number,
                extraction_method=ExtractionMethod.OCR,
                native_text=native_text,
                original_text=ocr_text.text,
                normalized_text=normalized,
                native_quality=native_quality,
                selected_quality=selected_quality,
                language=self._page_language(selected_quality),
                ocr_mean_confidence=ocr_text.mean_confidence,
            )
        except Exception as error:
            return PageRecord(
                source_file=metadata.source_file,
                document_id=metadata.document_id,
                document_version=metadata.document_version,
                file_hash=metadata.file_hash,
                page_number=page_number,
                extraction_method=ExtractionMethod.FAILED,
                native_text=native_text,
                original_text="",
                normalized_text="",
                native_quality=native_quality,
                selected_quality=measure_text_quality(""),
                language=self._page_language(native_quality),
                error=f"{type(error).__name__}: {error}",
            )

    def _configuration(
        self,
        *,
        token_counter: TokenCounter,
        page_limit: int | None,
    ) -> dict[str, Any]:
        return {
            "pipeline_version": PIPELINE_VERSION,
            "expected_language": self._expected_language,
            "page_limit": page_limit,
            "tokenizer_name": token_counter.name,
            "chunking": asdict(self._chunking),
        }

    def _duplicate_summary(
        self,
        report: dict[str, Any],
        *,
        source_file: str,
        document_id: str,
        document_version: int,
        document_type: str,
        source: str,
        configuration: dict[str, Any],
        pages_output: Path,
        chunks_output: Path,
        report_output: Path,
    ) -> IngestionSummary | None:
        if report.get("configuration") != configuration:
            return None
        document = report.get("document")
        counts = report.get("counts")
        if not isinstance(document, dict) or not isinstance(counts, dict):
            return None
        if document.get("document_id") != document_id:
            return None
        expected_metadata = {
            "document_version": document_version,
            "document_type": document_type,
            "source": source,
        }
        conflicts = {
            key: (document.get(key), value)
            for key, value in expected_metadata.items()
            if document.get(key) != value
        }
        if conflicts:
            raise ValueError(
                f"Document {document_id} was already ingested with different metadata: {conflicts}"
            )
        if not pages_output.is_file() or not chunks_output.is_file():
            return None
        return IngestionSummary(
            status=IngestionStatus.DUPLICATE,
            source_file=source_file,
            document_id=document_id,
            processed_pages=int(counts["processed_pages"]),
            native_pages=int(counts["native_pages"]),
            ocr_pages=int(counts["ocr_pages"]),
            failed_pages=int(counts["failed_pages"]),
            chunks=int(counts["chunks"]),
            pages_output=pages_output,
            chunks_output=chunks_output,
            report_output=report_output,
        )

    def ingest(
        self,
        pdf_path: Path,
        output_directory: Path,
        *,
        page_limit: int | None = None,
        document_version: int = 1,
        document_type: str = "unknown",
        source: str = "unknown",
    ) -> IngestionSummary:
        """Ingest one PDF or reuse identical completed artifacts."""

        if page_limit is not None and page_limit <= 0:
            raise ValueError("page_limit must be positive")
        if document_version <= 0:
            raise ValueError("document_version must be positive")
        if not document_type.strip() or not source.strip():
            raise ValueError("document_type and source must not be blank")

        validated = validate_pdf(
            pdf_path,
            maximum_file_size_bytes=self._maximum_pdf_bytes,
        )
        token_counter = self._get_token_counter()
        document_id = validated.file_hash
        metadata = DocumentMetadata(
            document_id=document_id,
            document_version=document_version,
            document_type=document_type,
            source=source,
            source_file=validated.path.name,
            file_hash=validated.file_hash,
        )
        configuration = self._configuration(
            token_counter=token_counter,
            page_limit=page_limit,
        )

        output_directory.mkdir(parents=True, exist_ok=True)
        output_stem = f"document-{validated.file_hash[:20]}"
        if page_limit is not None:
            output_stem = f"{output_stem}-first-{page_limit}"
        pages_output = output_directory / f"{output_stem}.pages.jsonl"
        chunks_output = output_directory / f"{output_stem}.chunks.jsonl"
        report_output = output_directory / f"{output_stem}.ingestion.json"
        duplicate = self._duplicate_summary(
            _read_json(report_output) or {},
            source_file=validated.path.name,
            document_id=document_id,
            document_version=document_version,
            document_type=document_type,
            source=source,
            configuration=configuration,
            pages_output=pages_output,
            chunks_output=chunks_output,
            report_output=report_output,
        )
        if duplicate is not None:
            return duplicate

        pages_temporary = pages_output.with_suffix(pages_output.suffix + ".tmp")
        chunks_temporary = chunks_output.with_suffix(chunks_output.suffix + ".tmp")
        native_pages = 0
        ocr_pages = 0
        failed_pages = 0
        chunk_count = 0
        processed_pages = 0
        structure_detector = LegalStructureDetector()

        try:
            with (
                validated.path.open("rb") as native_source,
                pymupdf.open(validated.path) as document,  # type: ignore[no-untyped-call]
                pages_temporary.open("w", encoding="utf-8", newline="\n") as pages_file,
                chunks_temporary.open("w", encoding="utf-8", newline="\n") as chunks_file,
            ):
                native_document = PdfReader(native_source, strict=False)
                native_page_count = len(native_document.pages)
                if document.page_count != native_page_count:
                    raise ValueError(
                        "PDF page-count mismatch: "
                        f"PyMuPDF={document.page_count}, pypdf={native_page_count}: "
                        f"{validated.path}"
                    )
                stop = document.page_count
                if page_limit is not None:
                    stop = min(stop, page_limit)

                for page_index in range(stop):
                    try:
                        native_text = native_document.pages[page_index].extract_text() or ""
                    except Exception:
                        native_text = ""
                    record = self._extract_page(
                        document[page_index],
                        native_text=native_text,
                        metadata=metadata,
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
                    spans = structure_detector.detect_page(record)
                    chunks = chunk_page_sections(
                        record,
                        spans,
                        metadata,
                        token_counter,
                        starting_index=chunk_count,
                        config=self._chunking,
                    )
                    for chunk in chunks:
                        chunks_file.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
                    chunk_count += len(chunks)
                    print(
                        f"{validated.path.name}: page {page_index + 1}/{stop} "
                        f"[{record.extraction_method.value}]"
                    )

            pages_temporary.replace(pages_output)
            chunks_temporary.replace(chunks_output)
        finally:
            pages_temporary.unlink(missing_ok=True)
            chunks_temporary.unlink(missing_ok=True)

        counts = {
            "processed_pages": processed_pages,
            "native_pages": native_pages,
            "ocr_pages": ocr_pages,
            "failed_pages": failed_pages,
            "chunks": chunk_count,
        }
        report = {
            "status": IngestionStatus.PROCESSED.value,
            "document": {
                "document_id": document_id,
                "document_version": document_version,
                "document_type": document_type,
                "source": source,
                "source_file": validated.path.name,
                "file_hash": validated.file_hash,
                "file_size_bytes": validated.file_size_bytes,
            },
            "configuration": configuration,
            "counts": counts,
            "outputs": {
                "pages": pages_output.name,
                "chunks": chunks_output.name,
                "report": report_output.name,
            },
        }
        _write_json(report_output, report)
        return IngestionSummary(
            status=IngestionStatus.PROCESSED,
            source_file=validated.path.name,
            document_id=document_id,
            processed_pages=processed_pages,
            native_pages=native_pages,
            ocr_pages=ocr_pages,
            failed_pages=failed_pages,
            chunks=chunk_count,
            pages_output=pages_output,
            chunks_output=chunks_output,
            report_output=report_output,
        )
