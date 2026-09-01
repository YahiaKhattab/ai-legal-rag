"""Qdrant client and collection management for the local vector store."""

from __future__ import annotations

from collections.abc import Sequence
from uuid import NAMESPACE_URL, UUID, uuid5

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PayloadSchemaType, PointStruct, VectorParams

from legal_rag.ingestion.models import ChunkRecord


class QdrantVectorStore:
    """Manage the local Qdrant collection used by the legal RAG system."""

    def __init__(
        self,
        *,
        url: str = "http://127.0.0.1:6333",
        api_key: str | None = None,
        collection_name: str = "legal_chunks",
        vector_size: int = 768,
        timeout: float = 60.0,
        client: QdrantClient | None = None,
    ) -> None:
        self._client = client or QdrantClient(url=url, api_key=api_key, timeout=timeout)
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

    # Payload fields filtered on in query/retriever.py's RetrievalFilters.
    # Qdrant requires an explicit index before a field can be used in a filter.
    _FILTERABLE_FIELDS = ("language", "document_type", "source", "document_id", "source_file")

    def create_collection(self) -> None:
        """Create the legal chunks collection if it does not exist, and make
        sure every field used for metadata filtering has a payload index.
        """

        collections = self._client.get_collections().collections
        existing_names = {collection.name for collection in collections}

        if self._collection_name not in existing_names:
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(
                    size=self._vector_size,
                    distance=Distance.COSINE,
                ),
            )

        self._ensure_payload_indexes()

    def _ensure_payload_indexes(self) -> None:
        """Create keyword payload indexes needed for metadata filtering.

        Safe to call repeatedly: re-creating an index that already exists
        with the same schema is a no-op on Qdrant's side, but we still
        guard each call in case of transient errors so one failing field
        doesn't stop the others from being indexed.
        """

        for field_name in self._FILTERABLE_FIELDS:
            try:
                self._client.create_payload_index(
                    collection_name=self._collection_name,
                    field_name=field_name,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            except Exception:  # noqa: BLE001 - index may already exist; safe to ignore
                pass

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

    def upsert_points(self, points: Sequence[PointStruct], batch_size: int = 64) -> None:
        """Insert or replace points in the Qdrant collection.

        Points are sent in small batches instead of one large request so
        that slow or unstable connections (e.g. to a cloud cluster) don't
        time out on a single oversized upload.
        """

        if not points:
            return

        points_list = list(points)

        for start in range(0, len(points_list), batch_size):
            batch = points_list[start : start + batch_size]
            self._client.upsert(
                collection_name=self._collection_name,
                points=batch,
                wait=True,
            )