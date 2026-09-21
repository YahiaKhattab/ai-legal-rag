"""Unit tests for legal_rag.vector_store.qdrant.QdrantVectorStore.

The real QdrantClient talks to a running Qdrant server over HTTP. We
patch QdrantClient itself so no server is needed; the goal is to test
QdrantVectorStore's own logic (deterministic point IDs, collection
creation guard, dimension validation) rather than the Qdrant client
library.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from uuid import UUID

import numpy as np
import pytest

from legal_rag.ingestion.models import (
    ChunkRecord,
    ExtractionMethod,
    LocatorType,
    SectionType,
    SourceFormat,
)
from legal_rag.vector_store.qdrant import QdrantVectorStore


def _chunk() -> ChunkRecord:
    return ChunkRecord(
        chunk_id="abc123",
        document_id="doc-1",
        document_version=1,
        chunk_index=0,
        original_text="text",
        normalized_text="text",
        section_type=SectionType.ARTICLE,
        section_title=None,
        page_start=1,
        page_end=1,
        source_format=SourceFormat.PDF,
        locator_type=LocatorType.PAGE,
        locator_start=1,
        locator_end=1,
        language="en",
        document_type="statute",
        source="tests",
        source_file="doc.pdf",
        file_hash="hash",
        extraction_methods=(ExtractionMethod.NATIVE,),
        original_start_char=0,
        original_end_char=4,
        token_count=1,
        tokenizer_name="fake",
        pipeline_version="1.0",
    )


@pytest.fixture
def fake_qdrant_client():
    with patch("legal_rag.vector_store.qdrant.QdrantClient") as client_cls:
        instance = MagicMock()
        client_cls.return_value = instance
        yield instance


def test_client_and_collection_name_properties(fake_qdrant_client):
    store = QdrantVectorStore(collection_name="legal_chunks")
    assert store.client is fake_qdrant_client
    assert store.collection_name == "legal_chunks"


def test_create_collection_creates_when_missing(fake_qdrant_client):
    existing = MagicMock()
    existing.name = "some_other_collection"
    fake_qdrant_client.get_collections.return_value.collections = [existing]

    store = QdrantVectorStore(collection_name="legal_chunks", vector_size=768)
    store.create_collection()

    fake_qdrant_client.create_collection.assert_called_once()
    _, kwargs = fake_qdrant_client.create_collection.call_args
    assert kwargs["collection_name"] == "legal_chunks"
    assert kwargs["vectors_config"].size == 768


def test_create_collection_skips_when_already_exists(fake_qdrant_client):
    existing = MagicMock()
    existing.name = "legal_chunks"
    fake_qdrant_client.get_collections.return_value.collections = [existing]

    store = QdrantVectorStore(collection_name="legal_chunks")
    store.create_collection()

    fake_qdrant_client.create_collection.assert_not_called()


def test_point_id_is_deterministic_for_the_same_chunk_id(fake_qdrant_client):
    store = QdrantVectorStore()
    id_a = store._point_id("chunk-1")
    id_b = store._point_id("chunk-1")
    assert id_a == id_b
    assert isinstance(id_a, UUID)


def test_point_id_differs_for_different_chunk_ids(fake_qdrant_client):
    store = QdrantVectorStore()
    assert store._point_id("chunk-1") != store._point_id("chunk-2")


def test_build_point_embeds_chunk_id_and_payload(fake_qdrant_client):
    store = QdrantVectorStore(vector_size=4)
    embedding = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)

    point = store.build_point(chunk=_chunk(), embedding=embedding)

    assert point.id == store._point_id("abc123")
    assert point.vector == pytest.approx([0.1, 0.2, 0.3, 0.4])
    assert point.payload["chunk_id"] == "abc123"


def test_build_point_rejects_non_1d_embedding(fake_qdrant_client):
    store = QdrantVectorStore(vector_size=4)
    embedding = np.zeros((2, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="one-dimensional"):
        store.build_point(chunk=_chunk(), embedding=embedding)


def test_build_point_rejects_wrong_dimension(fake_qdrant_client):
    store = QdrantVectorStore(vector_size=768)
    embedding = np.zeros(4, dtype=np.float32)

    with pytest.raises(ValueError, match="dimension must be 768"):
        store.build_point(chunk=_chunk(), embedding=embedding)


def test_upsert_points_calls_client_with_wait_true(fake_qdrant_client):
    store = QdrantVectorStore(collection_name="legal_chunks")
    fake_point = MagicMock()

    store.upsert_points([fake_point])

    fake_qdrant_client.upsert.assert_called_once_with(
        collection_name="legal_chunks", points=[fake_point], wait=True
    )


def test_upsert_points_skips_client_call_when_empty(fake_qdrant_client):
    store = QdrantVectorStore()
    store.upsert_points([])
    fake_qdrant_client.upsert.assert_not_called()
