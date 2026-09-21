"""Tests for persisted chunk JSONL reading."""

import json
from pathlib import Path

from legal_rag.embeddings.reader import read_chunks


def test_read_chunks(tmp_path: Path) -> None:
    path = tmp_path / "chunks.jsonl"

    record = {
        "chunk_id": "chunk-1",
        "document_id": "document-1",
        "document_version": 1,
        "chunk_index": 0,
        "original_text": "النص الأصلي",
        "normalized_text": "النص الاصلي",
        "section_type": "article",
        "section_title": "مادة",
        "page_start": 1,
        "page_end": 1,
        "source_format": "pdf",
        "locator_type": "page",
        "locator_start": 1,
        "locator_end": 1,
        "language": "ar",
        "document_type": "unknown",
        "source": "unknown",
        "source_file": "legal.pdf",
        "file_hash": "hash",
        "extraction_methods": ["ocr"],
        "original_start_char": 0,
        "original_end_char": 100,
        "token_count": 10,
        "tokenizer_name": "intfloat/multilingual-e5-base",
        "pipeline_version": "1.2.0",
    }

    path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")

    chunks = read_chunks(path)

    assert len(chunks) == 1
    assert chunks[0].chunk_id == "chunk-1"
    assert chunks[0].normalized_text == "النص الاصلي"
    assert chunks[0].extraction_methods[0].value == "ocr"


def test_read_chunks_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "chunks.jsonl"
    path.write_text("\n\n", encoding="utf-8")

    assert read_chunks(path) == []
