"""Keyword (BM25) retrieval, independent of Qdrant/dense search."""

from __future__ import annotations

from pathlib import Path

from legal_rag.ingestion.models import ChunkRecord
from legal_rag.query.bm25_index import BM25KeywordIndex, MetadataPredicate
from legal_rag.query.models import RetrievedChunk
from legal_rag.query.retriever import RetrievalFilters


class KeywordRetriever:
    """Top-K BM25 retrieval over a locally-built keyword index."""

    def __init__(self, index: BM25KeywordIndex | None = None) -> None:
        self._index = index or BM25KeywordIndex()

    @property
    def index(self) -> BM25KeywordIndex:
        return self._index

    def load(self, processed_directory: Path) -> int:
        """Build (or rebuild) the index from `*.chunks.jsonl` files.

        Needs to be re-run after new documents are ingested -- unlike
        Qdrant, there's no persistent server-side index here; it lives in
        this process's memory and reflects whatever was on disk the last
        time `load()` ran.
        """
        return self._index.build_from_directory(processed_directory)

    def search(
        self,
        query: str,
        top_k: int = 20,
        filters: RetrievalFilters | None = None,
    ) -> list[RetrievedChunk]:
        predicate = _build_metadata_predicate(filters) if filters else None
        hits = self._index.search(query, top_k=top_k, metadata_filter=predicate)

        return [
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                score=score,
                text=chunk.original_text or chunk.normalized_text,
                source_file=chunk.source_file,
                section_type=chunk.section_type,
                section_title=chunk.section_title,
                page=chunk.page_start,
                language=chunk.language,
                document_type=chunk.document_type,
                source=chunk.source,
                payload=chunk.to_dict(),
            )
            for chunk, score in hits
        ]


def _build_metadata_predicate(filters: RetrievalFilters) -> MetadataPredicate:
    def predicate(chunk: ChunkRecord) -> bool:
        if filters.language and chunk.language != filters.language:
            return False
        if filters.document_type and chunk.document_type != filters.document_type:
            return False
        if filters.source and chunk.source != filters.source:
            return False
        if filters.document_id and chunk.document_id != filters.document_id:
            return False
        return True

    return predicate
