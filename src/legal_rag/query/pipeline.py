"""End-to-end RAG answer pipeline: retrieve -> rerank -> prompt -> generate.

Closes the ingestion -> ... -> generation boundary documented as
"next stage" / "planned", and implements FR-003 (bilingual natural-language
search) and FR-004 (concise, cited AI summaries) together.
"""
from __future__ import annotations

from legal_rag.query.ollama_client import OllamaGenerationClient
from legal_rag.query.prompt_builder import build_grounded_prompt
from legal_rag.query.reranker import CrossEncoderReranker, get_default_reranker
from legal_rag.query.models import CitedAnswer
from legal_rag.query.retriever import LegalRetriever, RetrievalFilters


class RAGAnswerPipeline:
    def __init__(
        self,
        retriever: LegalRetriever,
        reranker: CrossEncoderReranker | None = None,
        generator: OllamaGenerationClient | None = None,
        retrieve_top_k: int = 20,
        rerank_top_n: int = 6,
    ) -> None:
        self._retriever = retriever
        self._reranker = reranker or get_default_reranker()
        self._generator = generator or OllamaGenerationClient()
        self._retrieve_top_k = retrieve_top_k
        self._rerank_top_n = rerank_top_n

    def answer(
        self,
        query: str,
        language: str = "mixed",
        filters: RetrievalFilters | None = None,
    ) -> CitedAnswer:
        retrieved = self._retriever.search(query, top_k=self._retrieve_top_k, filters=filters)

        if not retrieved:
            return CitedAnswer(
                query=query,
                answer_text=_no_evidence_message(language),
                language=language,
                citations=[],
                retrieved_chunk_ids=[],
            )

        reranked = self._reranker.rerank(query, retrieved, top_n=self._rerank_top_n)
        prompt, citations = build_grounded_prompt(query, reranked, language=language)
        answer_text = self._generator.generate(prompt)

        return CitedAnswer(
            query=query,
            answer_text=answer_text,
            language=language,
            citations=citations,
            retrieved_chunk_ids=[chunk.chunk_id for chunk in retrieved],
        )


def _no_evidence_message(language: str) -> str:
    if language == "ar":
        return "لم يتم العثور على مقاطع قانونية ذات صلة بالسؤال في قاعدة البيانات."
    return "No relevant legal excerpts were found in the indexed documents for this question."
