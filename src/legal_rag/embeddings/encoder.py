"""Local embedding encoder for multilingual legal text."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from legal_rag.embeddings.models import EmbeddingConfig


class EmbeddingEncoder:
    """Encode legal documents and queries using a local E5 model."""

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        """Initialize the local embedding model."""
        self.config = config or EmbeddingConfig()
        self._model = SentenceTransformer(
            self.config.model_name,
            device=self.config.device,
        )

    @property
    def dimension(self) -> int:
        """Return the dimensionality of generated embeddings."""
        dimension = self._model.get_sentence_embedding_dimension()

        if dimension is None:
            raise RuntimeError("Embedding model did not provide a dimension.")

        return dimension

    def encode_document(self, text: str) -> np.ndarray:
        """Encode a single document passage."""
        self._validate_text(text, "document")

        embeddings = self._encode([f"passage: {text}"])
        return np.asarray(embeddings[0], dtype=np.float32)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Encode multiple document passages as a batch."""
        for text in texts:
            self._validate_text(text, "document")

        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        prefixed_texts = [f"passage: {text}" for text in texts]
        return self._encode(prefixed_texts)

    def encode_query(self, text: str) -> np.ndarray:
        """Encode a single search query."""
        self._validate_text(text, "query")

        embeddings = self._encode([f"query: {text}"])
        return np.asarray(embeddings[0], dtype=np.float32)

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        """Encode prefixed texts and return normalized float32 vectors."""
        embeddings = self._model.encode(
            list(texts),
            normalize_embeddings=self.config.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        return np.asarray(embeddings, dtype=np.float32)

    @staticmethod
    def _validate_text(text: str, text_type: str) -> None:
        """Reject empty or whitespace-only text."""
        if not text.strip():
            raise ValueError(f"{text_type.capitalize()} text cannot be empty.")
