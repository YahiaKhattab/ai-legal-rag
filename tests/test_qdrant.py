import numpy as np
from qdrant_client import QdrantClient

from legal_rag.embeddings.models import EmbeddingConfig
from legal_rag.ingestion.models import (
    ChunkRecord,
    ExtractionMethod,
    LocatorType,
    SectionType,
    SourceFormat,
)
from legal_rag.vector_store.qdrant import QdrantVectorStore


def test_qdrant_collection_creation() -> None:
    store = QdrantVectorStore(client=QdrantClient(":memory:"))

    store.create_collection()

    collections = store.client.get_collections().collections
    names = {collection.name for collection in collections}

    assert store.collection_name in names


def test_build_point_preserves_chunk_metadata() -> None:
    store = QdrantVectorStore(client=QdrantClient(":memory:"))

    chunk = ChunkRecord(
        chunk_id="chunk-123",
        document_id="document-123",
        document_version=1,
        chunk_index=0,
        original_text="النص الأصلي",
        normalized_text="النص الأصلي",
        section_type=SectionType.ARTICLE,
        section_title="المادة الأولى",
        page_start=1,
        page_end=1,
        source_format=SourceFormat.PDF,
        locator_type=LocatorType.PAGE,
        locator_start=1,
        locator_end=1,
        language="ar",
        document_type="law",
        source="official",
        source_file="law.pdf",
        file_hash="hash-123",
        extraction_methods=(ExtractionMethod.OCR,),
        original_start_char=0,
        original_end_char=50,
        token_count=10,
        tokenizer_name=EmbeddingConfig().model_name,
        pipeline_version="1.2.0",
    )

    embedding = np.zeros(768, dtype=np.float32)

    point = store.build_point(
        chunk=chunk,
        embedding=embedding,
    )

    assert point.id is not None

    assert isinstance(point.vector, list)
    assert len(point.vector) == 768

    assert point.payload is not None
    assert point.payload["chunk_id"] == "chunk-123"
    assert point.payload["document_id"] == "document-123"
    assert point.payload["source_file"] == "law.pdf"
    assert point.payload["page_start"] == 1
    assert point.payload["section_title"] == "المادة الأولى"


def test_upsert_points() -> None:
    store = QdrantVectorStore(
        collection_name="test_legal_chunks",
        client=QdrantClient(":memory:"),
    )

    store.create_collection()

    chunk = ChunkRecord(
        chunk_id="chunk-upsert-123",
        document_id="document-123",
        document_version=1,
        chunk_index=0,
        original_text="النص الأصلي",
        normalized_text="النص الأصلي",
        section_type=SectionType.ARTICLE,
        section_title="المادة الأولى",
        page_start=1,
        page_end=1,
        source_format=SourceFormat.PDF,
        locator_type=LocatorType.PAGE,
        locator_start=1,
        locator_end=1,
        language="ar",
        document_type="law",
        source="official",
        source_file="law.pdf",
        file_hash="hash-123",
        extraction_methods=(ExtractionMethod.OCR,),
        original_start_char=0,
        original_end_char=50,
        token_count=10,
        tokenizer_name=EmbeddingConfig().model_name,
        pipeline_version="1.2.0",
    )

    embedding = np.zeros(768, dtype=np.float32)

    point = store.build_point(
        chunk=chunk,
        embedding=embedding,
    )

    store.upsert_points([point])

    result = store.client.retrieve(
        collection_name=store.collection_name,
        ids=[point.id],
        with_payload=True,
        with_vectors=True,
    )

    assert len(result) == 1
    assert str(result[0].id) == str(point.id)
    assert result[0].payload is not None
    assert result[0].payload["chunk_id"] == "chunk-upsert-123"
    assert result[0].payload["document_id"] == "document-123"
    assert result[0].payload["source_file"] == "law.pdf"
