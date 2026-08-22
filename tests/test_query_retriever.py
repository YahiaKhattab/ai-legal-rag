from typing import cast

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from legal_rag.query.query_embedder import QueryEmbedder
from legal_rag.query.retriever import LegalRetriever, RetrievalFilters
from legal_rag.vector_store.qdrant import QdrantVectorStore


class FakeQueryEmbedder:
    def encode_query(self, query: str) -> list[float]:
        assert query == "question"
        return [1.0, 0.0, 0.0]


def test_retriever_maps_payload_and_applies_source_filter() -> None:
    store = QdrantVectorStore(
        collection_name="retrieval-test",
        vector_size=3,
        client=QdrantClient(":memory:"),
    )
    store.create_collection()
    store.client.upsert(
        collection_name=store.collection_name,
        points=[
            PointStruct(
                id=1,
                vector=[1.0, 0.0, 0.0],
                payload={
                    "chunk_id": "chunk-1",
                    "document_id": "document-1",
                    "original_text": "relevant text",
                    "source": "official",
                    "source_file": "law.txt",
                    "page_start": 3,
                },
            ),
            PointStruct(
                id=2,
                vector=[1.0, 0.0, 0.0],
                payload={
                    "chunk_id": "chunk-2",
                    "document_id": "document-2",
                    "original_text": "excluded text",
                    "source": "other",
                },
            ),
        ],
        wait=True,
    )
    retriever = LegalRetriever(
        store=store,
        embedder=cast(QueryEmbedder, FakeQueryEmbedder()),
    )

    results = retriever.search(
        "question",
        filters=RetrievalFilters(source="official"),
    )

    assert len(results) == 1
    assert results[0].chunk_id == "chunk-1"
    assert results[0].text == "relevant text"
    assert results[0].page == 3
