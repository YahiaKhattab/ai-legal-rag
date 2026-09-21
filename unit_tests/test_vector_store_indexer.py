"""Unit tests for legal_rag.vector_store.indexer.QdrantIndexer.

QdrantIndexer coordinates read_chunks -> embed -> build_point -> upsert.
Every collaborator is injected as a fake that satisfies the module's own
EmbedderProtocol/VectorStoreProtocol, so this file tests QdrantIndexer's
own orchestration logic in isolation.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from legal_rag.vector_store.indexer import QdrantIndexer


class _FakeEmbedder:
    def __init__(self, dimension=4):
        self.dimension = dimension
        self.calls = []

    def embed_chunks(self, chunks):
        self.calls.append(chunks)
        return np.zeros((len(chunks), self.dimension), dtype=np.float32)


class _FakeStore:
    def __init__(self):
        self.created = False
        self.upserted_points = []
        self.build_point_calls = []

    def create_collection(self):
        self.created = True

    def build_point(self, *, chunk, embedding):
        self.build_point_calls.append((chunk, embedding))
        return f"point-for-{chunk.chunk_id}"

    def upsert_points(self, points):
        self.upserted_points.extend(points)


def _write_chunks_file(path, chunk_ids):
    import json

    with path.open("w", encoding="utf-8") as handle:
        for chunk_id in chunk_ids:
            record = {
                "chunk_id": chunk_id,
                "document_id": "doc-1",
                "document_version": 1,
                "chunk_index": 0,
                "original_text": "text",
                "normalized_text": "text",
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
                "original_end_char": 4,
                "token_count": 1,
                "tokenizer_name": "fake",
                "pipeline_version": "1.0",
            }
            handle.write(json.dumps(record) + "\n")


def test_ensure_collection_delegates_to_the_store():
    store = _FakeStore()
    indexer = QdrantIndexer(store=store, embedder=_FakeEmbedder())

    indexer.ensure_collection()

    assert store.created is True


def test_index_file_returns_zero_for_empty_chunks_file(tmp_path):
    path = tmp_path / "empty.chunks.jsonl"
    path.write_text("", encoding="utf-8")
    store = _FakeStore()
    embedder = _FakeEmbedder()

    total = QdrantIndexer(store=store, embedder=embedder).index_file(path)

    assert total == 0
    assert embedder.calls == []
    assert store.upserted_points == []


def test_index_file_embeds_and_upserts_all_chunks(tmp_path):
    path = tmp_path / "doc.chunks.jsonl"
    _write_chunks_file(path, ["chunk-1", "chunk-2"])
    store = _FakeStore()
    embedder = _FakeEmbedder()

    total = QdrantIndexer(store=store, embedder=embedder).index_file(path)

    assert total == 2
    assert len(store.build_point_calls) == 2
    assert store.upserted_points == ["point-for-chunk-1", "point-for-chunk-2"]


def test_index_directory_creates_collection_and_indexes_all_files(tmp_path):
    _write_chunks_file(tmp_path / "a.chunks.jsonl", ["chunk-1"])
    _write_chunks_file(tmp_path / "b.chunks.jsonl", ["chunk-2", "chunk-3"])
    (tmp_path / "irrelevant.txt").write_text("not a chunks file", encoding="utf-8")

    store = _FakeStore()
    embedder = _FakeEmbedder()

    total = QdrantIndexer(store=store, embedder=embedder).index_directory(tmp_path)

    assert total == 3
    assert store.created is True


def test_index_directory_with_no_matching_files_still_creates_collection(tmp_path):
    store = _FakeStore()
    total = QdrantIndexer(store=store, embedder=_FakeEmbedder()).index_directory(tmp_path)

    assert total == 0
    assert store.created is True


def test_default_constructor_builds_a_real_store_and_embedder(monkeypatch):
    """When no store/embedder is injected, QdrantIndexer should build the
    real QdrantVectorStore + BatchEmbedder(EmbeddingEncoder()) -- verified
    here by patching those constructors and checking they were used.
    """

    with patch("legal_rag.vector_store.indexer.QdrantVectorStore") as store_cls, patch(
        "legal_rag.vector_store.indexer.BatchEmbedder"
    ) as embedder_cls, patch("legal_rag.vector_store.indexer.EmbeddingEncoder") as encoder_cls:
        QdrantIndexer()

        store_cls.assert_called_once_with()
        encoder_cls.assert_called_once_with()
        embedder_cls.assert_called_once_with(encoder_cls.return_value)
