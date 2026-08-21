"""Vector similarity retrieval against the `legal_chunks` Qdrant collection.

Implements FR-003 ("search legal documents using natural language in both
Arabic and English") on top of the already-implemented Qdrant vector store
and embeddings.
"""
from __future__ import annotations

from dataclasses import dataclass

from legal_rag.query.chunk_text_store import ChunkTextStore
from legal_rag.query.models import RetrievedChunk
from legal_rag.query.query_embedder import QueryEmbedder, get_default_query_embedder
from legal_rag.vector_store.qdrant import QdrantVectorStore


@dataclass
class RetrievalFilters:
    """Optional metadata filters, applied as a Qdrant payload filter."""

    language: str | None = None  # "ar" | "en" | "mixed"
    document_type: str | None = None
    source: str | None = None
    document_id: str | None = None


class LegalRetriever:
    """Top-K semantic retrieval over indexed legal chunks."""

    def __init__(
        self,
        store: QdrantVectorStore | None = None,
        embedder: QueryEmbedder | None = None,
        collection_name: str = "legal_chunks",
        text_store: ChunkTextStore | None = None,
    ) -> None:
        self._store = store or QdrantVectorStore(collection_name=collection_name)
        self._embedder = embedder or get_default_query_embedder()
        # Optional fallback used only when the Qdrant payload doesn't carry
        # chunk text directly -- see chunk_text_store.py for why this exists.
        self._text_store = text_store

    def search(
        self,
        query: str,
        top_k: int = 20,
        filters: RetrievalFilters | None = None,
        score_threshold: float | None = None,
    ) -> list[RetrievedChunk]:
        """Embed `query` and return the top_k nearest chunks by cosine
        similarity. Qdrant already returns hits sorted by score.
        """
        vector = self._embedder.encode_query(query)
        qdrant_filter = _build_filter(filters) if filters else None

        results = self._store.client.query_points(
            collection_name=self._store.collection_name,
            query=vector,
            limit=top_k,
            query_filter=qdrant_filter,
            score_threshold=score_threshold,
            with_payload=True,
        )

        chunks: list[RetrievedChunk] = []
        for point in results.points:
            payload = point.payload or {}
            chunk_id = str(payload.get("chunk_id", point.id))

            text = payload.get("original_text") or payload.get("normalized_text") or payload.get("text") or ""
            if not text and self._text_store is not None:
                text = self._text_store.get_text(chunk_id) or ""

            chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    document_id=str(payload.get("document_id", "")),
                    score=float(point.score),
                    text=text,
                    source_file=payload.get("source_file"),
                    section_type=payload.get("section_type"),
                    section_title=payload.get("section_title"),
                    page=payload.get("page_start"),
                    language=payload.get("language"),
                    document_type=payload.get("document_type"),
                    source=payload.get("source"),
                    payload=payload,
                )
            )
        return chunks


def _build_filter(filters: RetrievalFilters):
    from qdrant_client import models as qmodels

    conditions = []
    if filters.language:
        conditions.append(
            qmodels.FieldCondition(key="language", match=qmodels.MatchValue(value=filters.language))
        )
    if filters.document_type:
        conditions.append(
            qmodels.FieldCondition(key="document_type", match=qmodels.MatchValue(value=filters.document_type))
        )
    if filters.source:
        conditions.append(qmodels.FieldCondition(key="source", match=qmodels.MatchValue(value=filters.source)))
    if filters.document_id:
        conditions.append(
            qmodels.FieldCondition(key="document_id", match=qmodels.MatchValue(value=filters.document_id))
        )
    if not conditions:
        return None
    return qmodels.Filter(must=conditions)
