"""Batch embedding operations for embedding-ready legal chunks."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from legal_rag.ingestion.models import ChunkRecord

from .encoder import EmbeddingEncoder


class BatchEmbedder:
    """Encode multiple embedding-ready chunks while preserving their order."""

    def __init__(self, encoder: EmbeddingEncoder) -> None:
        self._encoder = encoder

    def embed_chunks(
        self,
        chunks: Sequence[ChunkRecord],
    ) -> np.ndarray:
        """Return one embedding vector for each chunk in input order."""
        texts = [chunk.normalized_text for chunk in chunks]

        return self._encoder.encode_documents(texts)
