"""Index processed legal chunks into the local Qdrant vector store."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
from qdrant_client.models import PointStruct

from legal_rag.embeddings.batch import BatchEmbedder
from legal_rag.embeddings.encoder import EmbeddingEncoder
from legal_rag.embeddings.reader import read_chunks
from legal_rag.ingestion.models import ChunkRecord
from legal_rag.vector_store.qdrant import QdrantVectorStore


class EmbedderProtocol(Protocol):
    """Protocol required by the Qdrant indexer for embedding chunks."""

    def embed_chunks(
        self,
        chunks: list[ChunkRecord],
    ) -> np.ndarray:
        """Return one embedding vector for each chunk."""
        ...


class VectorStoreProtocol(Protocol):
    """Protocol required by the Qdrant indexer for vector storage."""

    def create_collection(self) -> None:
        """Create the target collection if it does not exist."""
        ...

    def build_point(
        self,
        *,
        chunk: ChunkRecord,
        embedding: np.ndarray,
    ) -> PointStruct:
        """Build a Qdrant point from a chunk and its embedding."""
        ...

    def upsert_points(self, points: list[PointStruct]) -> None:
        """Insert or replace points in the vector store."""
        ...


class QdrantIndexer:
    """Build embeddings for processed chunks and store them in Qdrant."""

    def __init__(
        self,
        *,
        store: VectorStoreProtocol | None = None,
        embedder: EmbedderProtocol | None = None,
    ) -> None:
        self._store = store or QdrantVectorStore()
        self._embedder = embedder or BatchEmbedder(EmbeddingEncoder())

    def ensure_collection(self) -> None:
        """Create the Qdrant collection if it does not already exist."""
        self._store.create_collection()

    def index_file(self, chunks_file: Path) -> int:
        """Index all chunks from one processed chunks file."""

        chunks = read_chunks(chunks_file)

        if not chunks:
            return 0

        embeddings = self._embedder.embed_chunks(chunks)

        points = [
            self._store.build_point(
                chunk=chunk,
                embedding=embedding,
            )
            for chunk, embedding in zip(
                chunks,
                embeddings,
                strict=True,
            )
        ]

        self._store.upsert_points(points)

        return len(points)

    def index_directory(self, processed_directory: Path) -> int:
        """Index all processed chunk files in a directory."""

        self.ensure_collection()

        total_chunks = 0

        for chunks_file in sorted(
            processed_directory.glob("*.chunks.jsonl")
        ):
            total_chunks += self.index_file(chunks_file)

        return total_chunks