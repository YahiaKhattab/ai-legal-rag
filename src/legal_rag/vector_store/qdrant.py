"""Qdrant client and collection management for the local vector store."""

from __future__ import annotations

from collections.abc import Sequence
from uuid import NAMESPACE_URL, UUID, uuid5

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from legal_rag.ingestion.models import ChunkRecord


class QdrantVectorStore:
    """Manage the local Qdrant collection used by the legal RAG system."""

    def __init__(
        self,
        *,
        url: str = "http://127.0.0.1:6333",
        collection_name: str = "legal_chunks",
        vector_size: int = 768,
        client: QdrantClient | None = None,
    ) -> None:
        self._client = client or QdrantClient(url=url)
        self._collection_name = collection_name
        self._vector_size = vector_size

    @property
    def client(self) -> QdrantClient:
        """Return the underlying Qdrant client."""

        return self._client

    @property
    def collection_name(self) -> str:
        """Return the configured collection name."""

        return self._collection_name

    def create_collection(self) -> None:
        """Create the legal chunks collection if it does not exist."""

        collections = self._client.get_collections().collections
        existing_names = {collection.name for collection in collections}

        if self._collection_name in existing_names:
            return

        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=VectorParams(
                size=self._vector_size,
                distance=Distance.COSINE,
            ),
        )

    @staticmethod
    def _point_id(chunk_id: str) -> UUID:
        """Create a deterministic UUID from the application chunk ID."""

        return uuid5(
            NAMESPACE_URL,
            f"legal-rag:{chunk_id}",
        )

    def build_point(
        self,
        *,
        chunk: ChunkRecord,
        embedding: np.ndarray,
    ) -> PointStruct:
        """Build a Qdrant point from one chunk and its embedding."""

        if embedding.ndim != 1:
            raise ValueError("embedding must be a one-dimensional vector")

        if embedding.shape[0] != self._vector_size:
            raise ValueError(
                f"embedding dimension must be {self._vector_size}, got {embedding.shape[0]}"
            )

        return PointStruct(
            id=self._point_id(chunk.chunk_id),
            vector=embedding.astype(np.float32).tolist(),
            payload=chunk.to_dict(),
        )

    def upsert_points(self, points: Sequence[PointStruct]) -> None:
        """Insert or replace points in the Qdrant collection."""

        if not points:
            return

        self._client.upsert(
            collection_name=self._collection_name,
            points=list(points),
            wait=True,
        )
