import json
from pathlib import Path
from typing import Any

import pymupdf

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
        json.loads(line)
        for line in summary.pages_output.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["extraction_method"] for record in records] == ["native", "ocr"]
    assert all(record["document_sha256"] for record in records)


def test_pipeline_corrects_native_digits_only_with_coordinate_evidence() -> None:
    pipeline = IngestionPipeline(expected_language="ar")

    record = pipeline._extract_page(
        RtlNativePage(),
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
        json.loads(line)
        for line in summary.pages_output.read_text(encoding="utf-8").splitlines()
    ]
    failed_record = page_records[1]

    assert failed_record["page_number"] == 2
    assert failed_record["extraction_method"] == "failed"
    assert "OCR output failed text-quality checks" in failed_record["error"]

    chunk_records = [
        json.loads(line)
        for line in summary.chunks_output.read_text(encoding="utf-8").splitlines()
    ]
    assert all(chunk["page_number"] != 2 for chunk in chunk_records)
