"""Unit tests for legal_rag.ingestion.pipeline.IngestionPipeline.

We exercise this against a real tiny .txt input (TXT ingestion has no
heavy dependencies) with a fake, injected TokenCounter so results are
deterministic and no real E5 tokenizer download is needed. This lets us
test the actual orchestration behavior end-to-end: file writing,
idempotency/duplicate detection, and input validation.
"""

from __future__ import annotations

import json

import pytest

from legal_rag.ingestion.models import IngestionStatus
from legal_rag.ingestion.pipeline import IngestionPipeline


class _OneTokenPerCharCounter:
    name = "fake-counter"

    def count_passage(self, text: str) -> int:
        return len(text)

    def count_content(self, text: str) -> int:
        return len(text)


def _pipeline() -> IngestionPipeline:
    return IngestionPipeline(expected_language="en", token_counter=_OneTokenPerCharCounter())


def _write_txt(tmp_path, text="Article 1: Introductory legal text for the unit test."):
    path = tmp_path / "law.txt"
    path.write_text(text, encoding="utf-8")
    return path


def test_ingest_produces_sources_chunks_and_report_files(tmp_path):
    document_path = _write_txt(tmp_path)
    output_dir = tmp_path / "out"

    summary = _pipeline().ingest(
        document_path, output_dir, document_type="statute", source="unit-tests"
    )

    assert summary.status is IngestionStatus.PROCESSED
    assert summary.processed_records == 1
    assert summary.chunks >= 1
    assert summary.sources_output.is_file()
    assert summary.chunks_output.is_file()
    assert summary.report_output.is_file()


def test_ingest_writes_valid_jsonl_source_and_chunk_records(tmp_path):
    document_path = _write_txt(tmp_path)
    output_dir = tmp_path / "out"

    summary = _pipeline().ingest(document_path, output_dir)

    source_lines = summary.sources_output.read_text(encoding="utf-8").splitlines()
    chunk_lines = summary.chunks_output.read_text(encoding="utf-8").splitlines()
    assert len(source_lines) == 1
    assert len(chunk_lines) >= 1
    parsed_source = json.loads(source_lines[0])
    assert parsed_source["document_id"] == summary.document_id


def test_ingest_second_call_is_reported_as_duplicate(tmp_path):
    document_path = _write_txt(tmp_path)
    output_dir = tmp_path / "out"
    pipeline = _pipeline()

    first = pipeline.ingest(document_path, output_dir, document_type="statute", source="unit-tests")
    second = pipeline.ingest(document_path, output_dir, document_type="statute", source="unit-tests")

    assert first.status is IngestionStatus.PROCESSED
    assert second.status is IngestionStatus.DUPLICATE
    assert second.document_id == first.document_id
    assert second.chunks == first.chunks


def test_ingest_same_document_id_but_different_metadata_raises(tmp_path):
    document_path = _write_txt(tmp_path)
    output_dir = tmp_path / "out"
    pipeline = _pipeline()

    pipeline.ingest(document_path, output_dir, document_type="statute", source="source-a")

    with pytest.raises(ValueError, match="already ingested with different metadata"):
        pipeline.ingest(document_path, output_dir, document_type="regulation", source="source-a")


def test_ingest_rejects_non_positive_document_version(tmp_path):
    document_path = _write_txt(tmp_path)
    with pytest.raises(ValueError, match="document_version must be positive"):
        _pipeline().ingest(document_path, tmp_path / "out", document_version=0)


def test_ingest_rejects_non_positive_page_limit(tmp_path):
    document_path = _write_txt(tmp_path)
    with pytest.raises(ValueError, match="page_limit must be positive"):
        _pipeline().ingest(document_path, tmp_path / "out", page_limit=0)


def test_ingest_rejects_blank_document_type_or_source(tmp_path):
    document_path = _write_txt(tmp_path)
    with pytest.raises(ValueError, match="must not be blank"):
        _pipeline().ingest(document_path, tmp_path / "out", document_type="   ", source="x")


def test_ingestion_summary_compatibility_aliases_match_underlying_fields(tmp_path):
    document_path = _write_txt(tmp_path)
    summary = _pipeline().ingest(document_path, tmp_path / "out")

    assert summary.processed_pages == summary.processed_records
    assert summary.native_pages == summary.direct_records
    assert summary.ocr_pages == summary.ocr_records
    assert summary.failed_pages == summary.failed_records
    assert summary.pages_output == summary.sources_output


def test_ingest_creates_output_directory_if_missing(tmp_path):
    document_path = _write_txt(tmp_path)
    output_dir = tmp_path / "does" / "not" / "exist"

    summary = _pipeline().ingest(document_path, output_dir)

    assert output_dir.is_dir()
    assert summary.sources_output.parent == output_dir
