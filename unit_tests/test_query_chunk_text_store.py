"""Unit tests for legal_rag.query.chunk_text_store.ChunkTextStore."""

from __future__ import annotations

import json

from legal_rag.query.chunk_text_store import ChunkTextStore


def _write_chunks_file(path, entries):
    """entries: list of (chunk_id, original_text, normalized_text)"""
    with path.open("w", encoding="utf-8") as handle:
        for chunk_id, original_text, normalized_text in entries:
            record = {
                "chunk_id": chunk_id,
                "document_id": "doc-1",
                "document_version": 1,
                "chunk_index": 0,
                "original_text": original_text,
                "normalized_text": normalized_text,
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
                "original_end_char": len(original_text),
                "token_count": 1,
                "tokenizer_name": "fake",
                "pipeline_version": "1.0",
            }
            handle.write(json.dumps(record) + "\n")


def test_get_text_returns_original_text_when_present(tmp_path):
    _write_chunks_file(tmp_path / "a.chunks.jsonl", [("chunk-1", "original", "normalized")])

    store = ChunkTextStore(tmp_path)

    assert store.get_text("chunk-1") == "original"


def test_get_text_falls_back_to_normalized_when_original_is_empty(tmp_path):
    _write_chunks_file(tmp_path / "a.chunks.jsonl", [("chunk-1", "", "normalized fallback")])

    store = ChunkTextStore(tmp_path)

    assert store.get_text("chunk-1") == "normalized fallback"


def test_get_text_returns_none_for_unknown_chunk_id(tmp_path):
    _write_chunks_file(tmp_path / "a.chunks.jsonl", [("chunk-1", "text", "text")])

    store = ChunkTextStore(tmp_path)

    assert store.get_text("does-not-exist") is None


def test_index_is_built_only_once_across_multiple_lookups(tmp_path):
    _write_chunks_file(tmp_path / "a.chunks.jsonl", [("chunk-1", "text", "text")])
    store = ChunkTextStore(tmp_path)

    store.get_text("chunk-1")
    # Delete the underlying file to prove the second lookup uses the
    # already-built in-memory index rather than re-scanning the directory.
    (tmp_path / "a.chunks.jsonl").unlink()

    assert store.get_text("chunk-1") == "text"


def test_indexes_across_multiple_chunk_files(tmp_path):
    _write_chunks_file(tmp_path / "a.chunks.jsonl", [("chunk-1", "from a", "from a")])
    _write_chunks_file(tmp_path / "b.chunks.jsonl", [("chunk-2", "from b", "from b")])

    store = ChunkTextStore(tmp_path)

    assert store.get_text("chunk-1") == "from a"
    assert store.get_text("chunk-2") == "from b"


def test_empty_directory_returns_none_for_any_lookup(tmp_path):
    store = ChunkTextStore(tmp_path)
    assert store.get_text("anything") is None
