"""Unit tests for legal_rag.embeddings.reader.read_chunks (JSONL parsing)."""

from __future__ import annotations

import json

import pytest

from legal_rag.embeddings.reader import read_chunks


def _valid_chunk_dict(**overrides) -> dict:
    base = {
        "chunk_id": "chunk-1",
        "document_id": "doc-1",
        "document_version": 1,
        "chunk_index": 0,
        "original_text": "Article 1: text",
        "normalized_text": "Article 1: text",
        "section_type": "article",
        "section_title": None,
        "page_start": 1,
        "page_end": 1,
        "source_format": "pdf",
        "locator_type": "page",
        "locator_start": 1,
        "locator_end": 1,
        "language": "en",
        "document_type": "statute",
        "source": "tests",
        "source_file": "doc.pdf",
        "file_hash": "hash",
        "extraction_methods": ["native"],
        "original_start_char": 0,
        "original_end_char": 15,
        "token_count": 5,
        "tokenizer_name": "fake",
        "pipeline_version": "1.0",
    }
    base.update(overrides)
    return base


def _write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def test_read_chunks_parses_a_single_valid_record(tmp_path):
    path = tmp_path / "chunks.jsonl"
    _write_jsonl(path, [_valid_chunk_dict()])

    chunks = read_chunks(path)

    assert len(chunks) == 1
    assert chunks[0].chunk_id == "chunk-1"
    assert chunks[0].extraction_methods[0].value == "native"


def test_read_chunks_skips_blank_lines(tmp_path):
    path = tmp_path / "chunks.jsonl"
    path.write_text(
        json.dumps(_valid_chunk_dict())
        + "\n\n   \n"
        + json.dumps(_valid_chunk_dict(chunk_id="chunk-2"))
        + "\n",
        encoding="utf-8",
    )

    chunks = read_chunks(path)

    assert [chunk.chunk_id for chunk in chunks] == ["chunk-1", "chunk-2"]


def test_read_chunks_raises_on_invalid_json(tmp_path):
    path = tmp_path / "chunks.jsonl"
    path.write_text("{not valid json\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSON on line 1"):
        read_chunks(path)


def test_read_chunks_raises_when_line_is_not_a_json_object(tmp_path):
    path = tmp_path / "chunks.jsonl"
    path.write_text("[1, 2, 3]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Expected a JSON object"):
        read_chunks(path)


def test_read_chunks_raises_on_missing_required_field(tmp_path):
    path = tmp_path / "chunks.jsonl"
    record = _valid_chunk_dict()
    del record["chunk_id"]
    _write_jsonl(path, [record])

    with pytest.raises(KeyError):
        read_chunks(path)


def test_read_chunks_returns_empty_list_for_empty_file(tmp_path):
    path = tmp_path / "chunks.jsonl"
    path.write_text("", encoding="utf-8")

    assert read_chunks(path) == []
