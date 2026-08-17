"""Query-side embedding for the E5 multilingual model.

The embedding stage documents that document text is embedded with the E5
"passage: " prefix. E5 models are trained with an asymmetric convention:
queries must use the "query: " prefix instead. Reusing the same model
weights keeps query and document vectors in the same space; only the
prefix differs. This is what makes natural-language Arabic/English/mixed
search (FR-003) work against the already-indexed passage vectors.
"""
from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer

_DEFAULT_MODEL_NAME = "intfloat/multilingual-e5-base"
_QUERY_PREFIX = "query: "


class QueryEmbedder:
    """Encodes a single natural-language query into the same 768-dim
    float32 vector space used by the document embeddings in Qdrant.
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL_NAME, device: str | None = None) -> None:
        self._model_name = model_name
        self._model = SentenceTransformer(model_name, device=device)

    def encode_query(self, query: str) -> list[float]:
        if not query or not query.strip():
            raise ValueError("Query text must not be empty.")
        prefixed = f"{_QUERY_PREFIX}{query.strip()}"
        vector = self._model.encode(
            prefixed,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vector.astype("float32").tolist()


@lru_cache(maxsize=1)
def get_default_query_embedder() -> QueryEmbedder:
    """Process-wide singleton so the model loads once per process, matching
    how the existing embedding encoder is used during ingestion.
    """
    return QueryEmbedder()
