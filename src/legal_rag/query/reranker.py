"""Cross-encoder reranking of retrieved candidates (Arabic/English aware).

The initial Qdrant vector search (bi-encoder) is fast but approximate --
query and document are embedded independently and compared by cosine
distance. Reranking re-scores the top-N candidates with a cross-encoder
that reads the query and each chunk together in a single forward pass,
which is slower per-pair but noticeably more accurate at judging
relevance. This is the standard reason to add a reranking stage after
initial retrieval, and matches "Improve candidate ordering after initial
vector retrieval" in the current status table.
"""

from __future__ import annotations

from functools import lru_cache

from sentence_transformers import CrossEncoder

from legal_rag.query.models import RerankedChunk, RetrievedChunk

# mMiniLM trained on mMARCO covers Arabic + English + ~100 languages and is
# small enough to run on CPU with reasonable latency alongside the existing
# CPU-based E5 embedding stack. Swap for a larger multilingual cross-encoder
# later if reranking quality needs to improve.
_DEFAULT_MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


class CrossEncoderReranker:
    def __init__(self, model_name: str = _DEFAULT_MODEL_NAME, device: str | None = None) -> None:
        self._model = CrossEncoder(model_name, device=device, max_length=512)

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_n: int = 6,
    ) -> list[RerankedChunk]:
        if not candidates:
            return []

        pairs = [(query, candidate.text) for candidate in candidates]
        scores = self._model.predict(pairs)

        reranked = [
            RerankedChunk(**candidate.__dict__, rerank_score=float(score))
            for candidate, score in zip(candidates, scores, strict=True)
        ]
        reranked.sort(key=lambda chunk: chunk.rerank_score, reverse=True)
        return reranked[:top_n]


@lru_cache(maxsize=1)
def get_default_reranker() -> CrossEncoderReranker:
    return CrossEncoderReranker()
