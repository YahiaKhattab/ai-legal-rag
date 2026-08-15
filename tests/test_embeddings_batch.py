"""Tests for batch embedding operations."""

from legal_rag.embeddings.batch import BatchEmbedder
from legal_rag.embeddings.encoder import EmbeddingEncoder
from legal_rag.ingestion.models import (
    ChunkRecord,
    ExtractionMethod,
    LocatorType,
    SectionType,
    SourceFormat,
)


def make_chunk(chunk_id: str, text: str) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        document_id="document-1",
        document_version=1,
        chunk_index=0,
        original_text=text,
        normalized_text=text,
        section_type=SectionType.ARTICLE,
        section_title="مادة",
        page_start=1,
        page_end=1,
        source_format=SourceFormat.PDF,
        locator_type=LocatorType.PAGE,
        locator_start=1,
        locator_end=1,
        language="ar",
        document_type="unknown",
        source="unknown",
        source_file="legal.pdf",
        file_hash="hash",
        extraction_methods=(ExtractionMethod.OCR,),
        original_start_char=0,
        original_end_char=len(text),
        token_count=10,
        tokenizer_name="intfloat/multilingual-e5-base",
        pipeline_version="1.2.0",
    )


def test_embed_chunks_preserves_order() -> None:
    encoder = EmbeddingEncoder()
    embedder = BatchEmbedder(encoder)

    chunks = [
        make_chunk("chunk-1", "شروط صحة العقد"),
        make_chunk("chunk-2", "أحكام حماية المستهلك"),
    ]

    embeddings = embedder.embed_chunks(chunks)

    assert embeddings.shape == (2, 768)
    assert embeddings.dtype.name == "float32"
    assert embeddings[0] @ embeddings[1] < 1.0


def test_embed_empty_chunks() -> None:
    encoder = EmbeddingEncoder()
    embedder = BatchEmbedder(encoder)

    embeddings = embedder.embed_chunks([])

    assert embeddings.shape == (0, 768)
    assert embeddings.dtype.name == "float32"
