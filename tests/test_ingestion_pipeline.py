import json
import shutil
from pathlib import Path
from typing import Any

import pymupdf
import pytest

import legal_rag.ingestion.extractors as extractors_module
from legal_rag.ingestion.models import (
    DocumentMetadata,
    ExtractionMethod,
    IngestionStatus,
    OcrText,
)
from legal_rag.ingestion.pipeline import IngestionPipeline


class WordTokenCounter:
    name = "test-word-counter"

    def count_passage(self, text: str) -> int:
        return len(text.split()) + 3

    def count_content(self, text: str) -> int:
        return len(text.split())


class FakeOcrEngine:
    def extract_page(self, page: Any) -> OcrText:
        del page
        return OcrText(text="Recognized legal text " * 10, mean_confidence=0.91)


class LowQualityOcrEngine:
    def extract_page(self, page: Any) -> OcrText:
        del page
        return OcrText(text="().(.", mean_confidence=0.56)


class NoDigitCoordinatePage:
    def get_text(self, output: str, *, sort: bool) -> dict[str, Any]:
        assert output == "rawdict"
        assert sort is False
        return {"blocks": []}


class MismatchedPdfReader:
    def __init__(self, stream: Any, *, strict: bool) -> None:
        del stream
        assert strict is False
        self.pages = [object()]


class RtlNativePage:
    def get_text(self, output: str, *, sort: bool) -> Any:
        assert sort is False
        if output == "rawdict":
            chars = [
                {"c": character, "bbox": (x, 0.0, x + 1.0, 1.0)}
                for character, x in zip("٤٢٠٢", (40.0, 30.0, 20.0, 10.0), strict=True)
            ]
            return {"blocks": [{"lines": [{"spans": [{"chars": chars}]}]}]}
        raise AssertionError(f"Unexpected output format: {output}")


def _metadata() -> DocumentMetadata:
    return DocumentMetadata(
        document_id="a" * 64,
        document_version=1,
        document_type="law",
        source="Official Gazette",
        source_file="law.pdf",
        file_hash="a" * 64,
    )


def _make_pdf(path: Path) -> None:
    with pymupdf.open() as document:  # type: ignore[no-untyped-call]
        native_page = document.new_page()
        native_page.insert_text((72, 72), "Native legal document text " * 10)
        document.new_page()
        document.save(path)


def _pipeline(*, ocr_engine: Any = None) -> IngestionPipeline:
    return IngestionPipeline(
        expected_language="en",
        ocr_engine=ocr_engine,
        token_counter=WordTokenCounter(),
    )


def test_pipeline_routes_pages_and_writes_full_contract(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    _make_pdf(pdf_path)

    summary = _pipeline(ocr_engine=FakeOcrEngine()).ingest(
        pdf_path,
        tmp_path / "processed",
        document_type="law",
        source="Test Authority",
    )

    assert summary.status is IngestionStatus.PROCESSED
    assert summary.native_pages == 1
    assert summary.ocr_pages == 1
    assert summary.failed_pages == 0
    page_records = [
        json.loads(line) for line in summary.pages_output.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["extraction_method"] for record in page_records] == ["native", "ocr"]
    assert page_records[0]["native_text"] == page_records[0]["original_text"]
    assert page_records[1]["native_text"] == ""
    assert page_records[1]["original_text"].startswith("Recognized legal text")

    chunk_records = [
        json.loads(line) for line in summary.chunks_output.read_text(encoding="utf-8").splitlines()
    ]
    required = {
        "chunk_id",
        "document_id",
        "document_version",
        "chunk_index",
        "original_text",
        "normalized_text",
        "section_title",
        "page_start",
        "page_end",
        "source_format",
        "locator_type",
        "locator_start",
        "locator_end",
        "language",
        "document_type",
        "source",
        "file_hash",
        "extraction_methods",
        "token_count",
        "pipeline_version",
    }
    assert required <= chunk_records[0].keys()
    assert [record["chunk_index"] for record in chunk_records] == list(range(len(chunk_records)))
    assert summary.report_output.is_file()


def test_pipeline_corrects_native_digits_without_mutating_original() -> None:
    pipeline = IngestionPipeline(expected_language="ar", token_counter=WordTokenCounter())
    native_text = ("قانون رقم ٠٢ لسنة ٤٢٠٢ ") * 8

    record = pipeline._extract_page(
        RtlNativePage(),
        native_text=native_text,
        metadata=_metadata(),
        page_number=1,
    )

    assert record.extraction_method is ExtractionMethod.NATIVE
    assert record.native_rtl_digit_correction_applied
    assert record.original_text == native_text
    assert "قانون رقم ٢٠ لسنة ٢٠٢٤" in record.normalized_text


def test_pipeline_rejects_low_quality_ocr_and_preserves_native_layer(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "low-quality-ocr.pdf"
    _make_pdf(pdf_path)

    summary = _pipeline(ocr_engine=LowQualityOcrEngine()).ingest(
        pdf_path,
        tmp_path / "processed",
    )

    assert summary.failed_pages == 1
    records = [
        json.loads(line) for line in summary.pages_output.read_text(encoding="utf-8").splitlines()
    ]
    failed = records[1]
    assert failed["native_text"] == ""
    assert failed["original_text"] == ""
    assert "OCR output failed text-quality checks" in failed["error"]


def test_pipeline_preserves_pypdf_logical_arabic_word_order() -> None:
    pipeline = IngestionPipeline(expected_language="ar", token_counter=WordTokenCounter())
    phrase = "المالية في غير مجلس"
    native_text = " ".join([phrase] * 8)

    record = pipeline._extract_page(
        NoDigitCoordinatePage(),
        native_text=native_text,
        metadata=_metadata(),
        page_number=1,
    )

    assert record.extraction_method is ExtractionMethod.NATIVE
    assert record.original_text == native_text
    assert not record.native_rtl_digit_correction_applied
    assert phrase in record.normalized_text


def test_duplicate_bytes_under_different_filename_are_reused(tmp_path: Path) -> None:
    first_path = tmp_path / "first.pdf"
    second_path = tmp_path / "renamed-copy.pdf"
    _make_pdf(first_path)
    shutil.copyfile(first_path, second_path)
    pipeline = _pipeline(ocr_engine=FakeOcrEngine())
    output = tmp_path / "processed"

    first = pipeline.ingest(first_path, output)
    first_ids = [
        json.loads(line)["chunk_id"]
        for line in first.chunks_output.read_text(encoding="utf-8").splitlines()
    ]
    second = pipeline.ingest(second_path, output)
    second_ids = [
        json.loads(line)["chunk_id"]
        for line in second.chunks_output.read_text(encoding="utf-8").splitlines()
    ]

    assert second.status is IngestionStatus.DUPLICATE
    assert first.document_id == second.document_id
    assert first.pages_output == second.pages_output
    assert first_ids == second_ids


def test_complete_reprocessing_produces_identical_chunk_ids(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    _make_pdf(pdf_path)
    pipeline = _pipeline(ocr_engine=FakeOcrEngine())
    output = tmp_path / "processed"

    first = pipeline.ingest(pdf_path, output)
    first_bytes = first.chunks_output.read_bytes()
    first.report_output.unlink()
    second = pipeline.ingest(pdf_path, output)

    assert second.status is IngestionStatus.PROCESSED
    assert second.chunks_output.read_bytes() == first_bytes


def test_pipeline_rejects_page_count_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_path = tmp_path / "mismatched.pdf"
    _make_pdf(pdf_path)
    monkeypatch.setattr(extractors_module, "PdfReader", MismatchedPdfReader)

    with pytest.raises(ValueError, match="page-count mismatch"):
        _pipeline(ocr_engine=FakeOcrEngine()).ingest(pdf_path, tmp_path / "processed")
