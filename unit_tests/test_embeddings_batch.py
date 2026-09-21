"""Unit tests for legal_rag.embeddings.batch.BatchEmbedder."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from legal_rag.embeddings.batch import BatchEmbedder
from legal_rag.ingestion.models import (
    ChunkRecord,
    ExtractionMethod,
    LocatorType,
    SectionType,
    SourceFormat,
)


def _chunk(index: int, normalized_text: str) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=f"chunk-{index}",
        document_id="doc-1",
        document_version=1,
        chunk_index=index,
        original_text=normalized_text,
        normalized_text=normalized_text,
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
        original_end_char=len(normalized_text),
        token_count=1,
        tokenizer_name="fake",
        pipeline_version="1.0",
    )


def test_embed_chunks_passes_normalized_text_in_order():
    fake_encoder = MagicMock()
    fake_encoder.encode_documents.return_value = np.zeros((2, 4), dtype=np.float32)

    embedder = BatchEmbedder(fake_encoder)
    chunks = [_chunk(0, "first chunk"), _chunk(1, "second chunk")]

    result = embedder.embed_chunks(chunks)

    fake_encoder.encode_documents.assert_called_once_with(["first chunk", "second chunk"])
    assert result.shape == (2, 4)


def test_embed_chunks_with_empty_list_calls_encoder_with_empty_list():
    fake_encoder = MagicMock()
    fake_encoder.encode_documents.return_value = np.zeros((0, 4), dtype=np.float32)

    embedder = BatchEmbedder(fake_encoder)
    result = embedder.embed_chunks([])

    fake_encoder.encode_documents.assert_called_once_with([])
    assert result.shape == (0, 4)
