import json
from pathlib import Path
from typing import Any

import pymupdf
import pytest

import legal_rag.ingestion.pipeline as pipeline_module
from legal_rag.ingestion.models import ExtractionMethod, OcrText
from legal_rag.ingestion.pipeline import IngestionPipeline


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
        if output == "text":
            return "قانون رقم ٠٢ لسنة ٤٢٠٢ " * 8
        if output == "rawdict":
            chars = [
                {"c": character, "bbox": (x, 0.0, x + 1.0, 1.0)}
                for character, x in zip("٤٢٠٢", (40.0, 30.0, 20.0, 10.0), strict=True)
            ]
            return {"blocks": [{"lines": [{"spans": [{"chars": chars}]}]}]}
        raise AssertionError(f"Unexpected output format: {output}")


def _make_pdf(path: Path) -> None:
    with pymupdf.open() as document:  # type: ignore[no-untyped-call]
        native_page = document.new_page()
        native_page.insert_text((72, 72), "Native legal document text " * 10)
        document.new_page()
        document.save(path)


def test_pipeline_routes_pages_and_writes_jsonl(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    _make_pdf(pdf_path)
    pipeline = IngestionPipeline(expected_language="en", ocr_engine=FakeOcrEngine())

    summary = pipeline.ingest(pdf_path, tmp_path / "processed")

    assert summary.native_pages == 1
    assert summary.ocr_pages == 1
    assert summary.failed_pages == 0
    records = [
        json.loads(line) for line in summary.pages_output.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["extraction_method"] for record in records] == ["native", "ocr"]
    assert all(record["document_sha256"] for record in records)


def test_pipeline_corrects_native_digits_only_with_coordinate_evidence() -> None:
    pipeline = IngestionPipeline(expected_language="ar")

    record = pipeline._extract_page(
        RtlNativePage(),
        native_text=(
            "\u0642\u0627\u0646\u0648\u0646 \u0631\u0642\u0645 \u0660\u0662 "
            "\u0644\u0633\u0646\u0629 \u0664\u0662\u0660\u0662 "
        )
        * 8,
        source_file="law.pdf",
        document_sha256="a" * 64,
        page_number=1,
    )

    assert record.extraction_method is ExtractionMethod.NATIVE
    assert record.native_rtl_digit_correction_applied
    assert "قانون رقم ٢٠ لسنة ٢٠٢٤" in record.normalized_text


def test_pipeline_rejects_low_quality_ocr_output(tmp_path: Path) -> None:
    pdf_path = tmp_path / "low-quality-ocr.pdf"
    _make_pdf(pdf_path)
    pipeline = IngestionPipeline(
        expected_language="en",
        ocr_engine=LowQualityOcrEngine(),
    )

    summary = pipeline.ingest(pdf_path, tmp_path / "processed")

    assert summary.native_pages == 1
    assert summary.ocr_pages == 0
    assert summary.failed_pages == 1

    page_records = [
        json.loads(line) for line in summary.pages_output.read_text(encoding="utf-8").splitlines()
    ]
    failed_record = page_records[1]

    assert failed_record["page_number"] == 2
    assert failed_record["extraction_method"] == "failed"
    assert "OCR output failed text-quality checks" in failed_record["error"]

    chunk_records = [
        json.loads(line) for line in summary.chunks_output.read_text(encoding="utf-8").splitlines()
    ]
    assert all(chunk["page_number"] != 2 for chunk in chunk_records)


def test_pipeline_preserves_pypdf_logical_arabic_word_order() -> None:
    pipeline = IngestionPipeline(expected_language="ar")
    phrase = (
        "\u0627\u0644\u0645\u0627\u0644\u064a\u0629 \u0641\u064a "
        "\u063a\u064a\u0631 \u0645\u062c\u0644\u0633"
    )
    native_text = " ".join([phrase] * 8)

    record = pipeline._extract_page(
        NoDigitCoordinatePage(),
        native_text=native_text,
        source_file="financial-law.pdf",
        document_sha256="b" * 64,
        page_number=1,
    )

    assert record.extraction_method is ExtractionMethod.NATIVE
    assert record.raw_text == native_text
    assert not record.native_rtl_digit_correction_applied
    assert phrase in record.normalized_text


def test_pipeline_rejects_page_count_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_path = tmp_path / "mismatched.pdf"
    _make_pdf(pdf_path)
    pipeline = IngestionPipeline(expected_language="en", ocr_engine=FakeOcrEngine())
    monkeypatch.setattr(pipeline_module, "PdfReader", MismatchedPdfReader)

    with pytest.raises(ValueError, match="page-count mismatch"):
        pipeline.ingest(pdf_path, tmp_path / "processed")
