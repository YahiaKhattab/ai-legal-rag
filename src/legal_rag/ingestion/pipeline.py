"""Document orchestration, idempotency, and atomic JSON persistence."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from legal_rag.ingestion.chunking import (
    DEFAULT_CHUNKING_CONFIG,
    PIPELINE_VERSION,
    chunk_source_sections,
)
from legal_rag.ingestion.extractors import (
    DocumentExtractor,
    DocxExtractor,
    OcrLanguage,
    PdfExtractor,
    TxtExtractor,
)
from legal_rag.ingestion.models import (
    ChunkingConfig,
    DocumentMetadata,
    ExtractionMethod,
    IngestionStatus,
    SourceFormat,
    SourceRecord,
)
from legal_rag.ingestion.ocr import OcrEngine, PaddleOcrEngine
from legal_rag.ingestion.quality import ExpectedLanguage
from legal_rag.ingestion.structure import LegalStructureDetector
from legal_rag.ingestion.tokenization import (
    DEFAULT_E5_REVISION,
    DEFAULT_E5_TOKENIZER,
    E5TokenCounter,
    TokenCounter,
)
from legal_rag.ingestion.validation import DEFAULT_MAXIMUM_DOCUMENT_BYTES, validate_document


@dataclass(frozen=True, slots=True)
class IngestionSummary:
    """Compact result returned to the CLI and callers."""

    status: IngestionStatus
    source_file: str
    document_id: str
    processed_records: int
    direct_records: int
    ocr_records: int
    failed_records: int
    chunks: int
    sources_output: Path
    chunks_output: Path
    report_output: Path

    @property
    def processed_pages(self) -> int:
        """Compatibility alias for PDF-era callers."""

        return self.processed_records

    @property
    def native_pages(self) -> int:
        """Compatibility alias for PDF-era callers."""

        return self.direct_records

    @property
    def ocr_pages(self) -> int:
        """Compatibility alias for PDF-era callers."""

        return self.ocr_records

    @property
    def failed_pages(self) -> int:
        """Compatibility alias for PDF-era callers."""

        return self.failed_records

    @property
    def pages_output(self) -> Path:
        """Compatibility alias for the former pages JSONL artifact."""

        return self.sources_output


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
        expected_language: ExpectedLanguage = "auto",
        ocr_engine: OcrEngine | None = None,
        token_counter: TokenCounter | None = None,
        tokenizer_identifier: str = DEFAULT_E5_TOKENIZER,
        tokenizer_revision: str = DEFAULT_E5_REVISION,
        chunking: ChunkingConfig = DEFAULT_CHUNKING_CONFIG,
        maximum_document_bytes: int = DEFAULT_MAXIMUM_DOCUMENT_BYTES,
    ) -> None:
        self._expected_language = expected_language
        self._ocr_engines: dict[OcrLanguage, OcrEngine] = {}
        if ocr_engine is not None:
            self._ocr_engines = {"ar": ocr_engine, "en": ocr_engine}
        self._token_counter = token_counter
        self._tokenizer_identifier = tokenizer_identifier
        self._tokenizer_revision = tokenizer_revision
        self._chunking = chunking
        self._maximum_document_bytes = maximum_document_bytes

    def _get_ocr_engine(self, language: OcrLanguage) -> OcrEngine:
        if language not in self._ocr_engines:
            self._ocr_engines[language] = PaddleOcrEngine(language=language)
        return self._ocr_engines[language]

    def _get_token_counter(self) -> TokenCounter:
        if self._token_counter is None:
            self._token_counter = E5TokenCounter.load(
                self._tokenizer_identifier,
                revision=self._tokenizer_revision,
            )
        return self._token_counter

    def _extract_page(
        self,
        page: Any,
        *,
        native_text: str,
        metadata: DocumentMetadata,
        page_number: int,
    ) -> SourceRecord:
        """Compatibility entry point delegated to the PDF extractor."""

        return self._pdf_extractor().extract_page(
            page,
            native_text=native_text,
            metadata=metadata,
            page_number=page_number,
        )

    def _pdf_extractor(self) -> PdfExtractor:
        return PdfExtractor(
            expected_language=self._expected_language,
            ocr_engine_factory=self._get_ocr_engine,
        )

    def _extractor_for(self, source_format: SourceFormat) -> DocumentExtractor:
        if source_format is SourceFormat.PDF:
            return self._pdf_extractor()
        if source_format is SourceFormat.DOCX:
            return DocxExtractor(expected_language=self._expected_language)
        return TxtExtractor(expected_language=self._expected_language)

    def _configuration(
        self,
        *,
        token_counter: TokenCounter,
        page_limit: int | None,
        source_format: SourceFormat,
    ) -> dict[str, Any]:
        return {
            "pipeline_version": PIPELINE_VERSION,
            "expected_language": self._expected_language,
            "page_limit": page_limit,
            "source_format": source_format.value,
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
        sources_output: Path,
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
        if not sources_output.is_file() or not chunks_output.is_file():
            return None
        return IngestionSummary(
            status=IngestionStatus.DUPLICATE,
            source_file=source_file,
            document_id=document_id,
            processed_records=int(counts["processed_records"]),
            direct_records=int(counts["direct_records"]),
            ocr_records=int(counts["ocr_records"]),
            failed_records=int(counts["failed_records"]),
            chunks=int(counts["chunks"]),
            sources_output=sources_output,
            chunks_output=chunks_output,
            report_output=report_output,
        )

    def ingest(
        self,
        document_path: Path,
        output_directory: Path,
        *,
        page_limit: int | None = None,
        document_version: int = 1,
        document_type: str = "unknown",
        source: str = "unknown",
    ) -> IngestionSummary:
        """Ingest one PDF, DOCX, or TXT file or reuse completed artifacts."""

        if page_limit is not None and page_limit <= 0:
            raise ValueError("page_limit must be positive")
        if document_version <= 0:
            raise ValueError("document_version must be positive")
        if not document_type.strip() or not source.strip():
            raise ValueError("document_type and source must not be blank")

        validated = validate_document(
            document_path,
            maximum_file_size_bytes=self._maximum_document_bytes,
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
            source_format=validated.source_format,
        )
        configuration = self._configuration(
            token_counter=token_counter,
            page_limit=page_limit,
            source_format=validated.source_format,
        )

        output_directory.mkdir(parents=True, exist_ok=True)
        output_stem = f"document-{validated.file_hash[:20]}"
        if page_limit is not None:
            output_stem = f"{output_stem}-first-{page_limit}"
        sources_output = output_directory / f"{output_stem}.sources.jsonl"
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
            sources_output=sources_output,
            chunks_output=chunks_output,
            report_output=report_output,
        )
        if duplicate is not None:
            return duplicate

        sources_temporary = sources_output.with_suffix(sources_output.suffix + ".tmp")
        chunks_temporary = chunks_output.with_suffix(chunks_output.suffix + ".tmp")
        direct_records = 0
        ocr_records = 0
        failed_records = 0
        chunk_count = 0
        processed_records = 0
        structure_detector = LegalStructureDetector()
        extractor = self._extractor_for(validated.source_format)

        try:
            with (
                sources_temporary.open("w", encoding="utf-8", newline="\n") as sources_file,
                chunks_temporary.open("w", encoding="utf-8", newline="\n") as chunks_file,
            ):
                for record in extractor.extract(
                    validated.path,
                    metadata,
                    page_limit=page_limit,
                ):
                    processed_records += 1
                    if record.extraction_method is ExtractionMethod.OCR:
                        ocr_records += 1
                    elif record.extraction_method is ExtractionMethod.FAILED:
                        failed_records += 1
                    else:
                        direct_records += 1

                    sources_file.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
                    spans = structure_detector.detect_source(record)
                    chunks = chunk_source_sections(
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
                        f"{validated.path.name}: {record.locator_type.value} "
                        f"{record.locator_start}-{record.locator_end} "
                        f"[{record.extraction_method.value}]"
                    )

            sources_temporary.replace(sources_output)
            chunks_temporary.replace(chunks_output)
        finally:
            sources_temporary.unlink(missing_ok=True)
            chunks_temporary.unlink(missing_ok=True)

        counts = {
            "processed_records": processed_records,
            "direct_records": direct_records,
            "ocr_records": ocr_records,
            "failed_records": failed_records,
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
                "source_format": validated.source_format.value,
            },
            "configuration": configuration,
            "counts": counts,
            "outputs": {
                "sources": sources_output.name,
                "chunks": chunks_output.name,
                "report": report_output.name,
            },
        }
        _write_json(report_output, report)
        return IngestionSummary(
            status=IngestionStatus.PROCESSED,
            source_file=validated.path.name,
            document_id=document_id,
            processed_records=processed_records,
            direct_records=direct_records,
            ocr_records=ocr_records,
            failed_records=failed_records,
            chunks=chunk_count,
            sources_output=sources_output,
            chunks_output=chunks_output,
            report_output=report_output,
        )
