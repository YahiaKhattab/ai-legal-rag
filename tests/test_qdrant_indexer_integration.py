import os
from pathlib import Path

import pytest

from legal_rag.embeddings.batch import BatchEmbedder
from legal_rag.embeddings.encoder import EmbeddingEncoder
from legal_rag.vector_store.indexer import QdrantIndexer
from legal_rag.vector_store.qdrant import QdrantVectorStore

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("LEGAL_RAG_RUN_INTEGRATION") != "1",
        reason="set LEGAL_RAG_RUN_INTEGRATION=1 to run local-service integration tests",
    ),
]


def test_index_processed_file_into_qdrant() -> None:
    processed_file = Path("data/processed/document-6e01d55aa22603bb8459.chunks.jsonl")

    store = QdrantVectorStore(
        collection_name="test_legal_rag_indexer",
    )

    store.create_collection()

    embedder = BatchEmbedder(
        EmbeddingEncoder(),
    )

    indexer = QdrantIndexer(
        store=store,
        embedder=embedder,
    )

    total = indexer.index_file(processed_file)

    assert total == 3

    result = store.client.count(
        collection_name=store.collection_name,
        exact=True,
    )

    assert result.count == 3
