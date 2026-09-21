"""Unit tests for legal_rag.ingestion.cli.

`_expand_inputs` is pure and tested directly against a real temp
directory. `main()` orchestrates IngestionPipeline + QdrantIndexer, so we
patch both classes (imported into the cli module) to avoid touching a
real Qdrant instance or doing real document extraction.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from legal_rag.ingestion import cli
from legal_rag.ingestion.models import IngestionStatus


# ------------------------------------------------------------- _expand_inputs


def test_expand_inputs_returns_single_file_as_is(tmp_path):
    file_path = tmp_path / "law.pdf"
    file_path.write_bytes(b"")

    result = cli._expand_inputs([file_path])

    assert result == [file_path]


def test_expand_inputs_expands_directory_to_supported_files_only(tmp_path):
    (tmp_path / "a.pdf").write_bytes(b"")
    (tmp_path / "b.docx").write_bytes(b"")
    (tmp_path / "c.txt").write_bytes(b"")
    (tmp_path / "notes.md").write_bytes(b"")  # unsupported, should be excluded

    result = cli._expand_inputs([tmp_path])

    names = sorted(path.name for path in result)
    assert names == ["a.pdf", "b.docx", "c.txt"]


def test_expand_inputs_mixes_files_and_directories(tmp_path):
    directory = tmp_path / "batch"
    directory.mkdir()
    (directory / "x.txt").write_bytes(b"")
    single_file = tmp_path / "single.pdf"
    single_file.write_bytes(b"")

    result = cli._expand_inputs([single_file, directory])

    assert single_file in result
    assert any(path.name == "x.txt" for path in result)


def test_expand_inputs_empty_directory_returns_empty_list(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    assert cli._expand_inputs([empty_dir]) == []


# ------------------------------------------------------------------------ main


def _fake_summary(**overrides):
    from pathlib import Path

    defaults = dict(
        status=IngestionStatus.PROCESSED,
        source_file="law.txt",
        document_id="abcdef0123456789",
        processed_records=1,
        direct_records=1,
        ocr_records=0,
        failed_records=0,
        chunks=3,
        sources_output=Path("out/doc.sources.jsonl"),
        chunks_output=Path("out/doc.chunks.jsonl"),
        report_output=Path("out/doc.ingestion.json"),
    )
    defaults.update(overrides)
    from legal_rag.ingestion.pipeline import IngestionSummary

    return IngestionSummary(**defaults)


def test_main_returns_zero_when_no_failures(tmp_path, monkeypatch):
    doc = tmp_path / "law.txt"
    doc.write_text("Article 1.", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["legal-rag-ingest", str(doc)])

    fake_pipeline = MagicMock()
    fake_pipeline.ingest.return_value = _fake_summary()
    fake_indexer = MagicMock()
    fake_indexer.index_file.return_value = 3

    with patch.object(cli, "IngestionPipeline", return_value=fake_pipeline), patch.object(
        cli, "QdrantIndexer", return_value=fake_indexer
    ):
        exit_code = cli.main()

    assert exit_code == 0
    fake_indexer.ensure_collection.assert_called_once()
    fake_indexer.index_file.assert_called_once()


def test_main_returns_one_when_ingestion_reports_failed_records(tmp_path, monkeypatch):
    doc = tmp_path / "law.txt"
    doc.write_text("Article 1.", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["legal-rag-ingest", str(doc)])

    fake_pipeline = MagicMock()
    fake_pipeline.ingest.return_value = _fake_summary(failed_records=1)
    fake_indexer = MagicMock()

    with patch.object(cli, "IngestionPipeline", return_value=fake_pipeline), patch.object(
        cli, "QdrantIndexer", return_value=fake_indexer
    ):
        exit_code = cli.main()

    assert exit_code == 1


def test_main_returns_one_when_ingestion_raises(tmp_path, monkeypatch):
    doc = tmp_path / "law.txt"
    doc.write_text("Article 1.", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["legal-rag-ingest", str(doc)])

    fake_pipeline = MagicMock()
    fake_pipeline.ingest.side_effect = ValueError("boom")
    fake_indexer = MagicMock()

    with patch.object(cli, "IngestionPipeline", return_value=fake_pipeline), patch.object(
        cli, "QdrantIndexer", return_value=fake_indexer
    ):
        exit_code = cli.main()

    assert exit_code == 1


def test_main_skips_indexing_when_no_chunks_were_produced(tmp_path, monkeypatch):
    doc = tmp_path / "law.txt"
    doc.write_text("Article 1.", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["legal-rag-ingest", str(doc)])

    fake_pipeline = MagicMock()
    fake_pipeline.ingest.return_value = _fake_summary(chunks=0)
    fake_indexer = MagicMock()

    with patch.object(cli, "IngestionPipeline", return_value=fake_pipeline), patch.object(
        cli, "QdrantIndexer", return_value=fake_indexer
    ):
        cli.main()

    fake_indexer.index_file.assert_not_called()


def test_main_raises_systemexit_when_no_supported_files_found(tmp_path, monkeypatch):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    monkeypatch.setattr("sys.argv", ["legal-rag-ingest", str(empty_dir)])

    with pytest.raises(SystemExit, match="No supported"):
        cli.main()


def test_main_rejects_pages_option_when_non_pdf_present(tmp_path, monkeypatch):
    doc = tmp_path / "law.txt"
    doc.write_text("Article 1.", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["legal-rag-ingest", str(doc), "--pages", "1"])

    with pytest.raises(SystemExit, match="--pages can only be used"):
        cli.main()
