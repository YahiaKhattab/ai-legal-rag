"""End-to-end grounded legal RAG answer pipeline.

Pipeline:
retrieve -> diversify -> rerank -> select evidence -> prompt -> generate
-> validate -> attach citations.
"""

from __future__ import annotations

from collections import defaultdict

from legal_rag.query.answer_validator import validate_numeric_claims
from legal_rag.query.models import (
    CitedAnswer,
    LegalExcerpt,
    RetrievedChunk,
)
from legal_rag.query.ollama_client import OllamaGenerationClient
from legal_rag.query.prompt_builder import build_grounded_prompt
from legal_rag.query.reranker import (
    CrossEncoderReranker,
    get_default_reranker,
)
from legal_rag.query.retriever import LegalRetriever, RetrievalFilters


class RAGAnswerPipeline:
    def __init__(
        self,
        retriever: LegalRetriever,
        reranker: CrossEncoderReranker | None = None,
        generator: OllamaGenerationClient | None = None,
        retrieve_top_k: int = 20,
        rerank_top_n: int = 6,
        evidence_top_n: int = 4,
    ) -> None:
        self._retriever = retriever
        self._reranker = reranker or get_default_reranker()
        self._generator = generator or OllamaGenerationClient()

        self._retrieve_top_k = retrieve_top_k
        self._rerank_top_n = rerank_top_n
        self._evidence_top_n = evidence_top_n

    def answer(
        self,
        query: str,
        language: str = "mixed",
        filters: RetrievalFilters | None = None,
    ) -> CitedAnswer:
        # ---------------------------------------------------------
        # 1. Retrieve candidate chunks from Qdrant.
        # ---------------------------------------------------------
        retrieved = self._retriever.search(
            query,
            top_k=self._retrieve_top_k,
            filters=filters,
        )

        if not retrieved:
            return CitedAnswer(
                query=query,
                answer_text=_no_evidence_message(language),
                language=language,
                citations=[],
                retrieved_chunk_ids=[],
                legal_excerpts=[],
            )

        # ---------------------------------------------------------
        # 2. Diversify candidates by document.
        #
        # Qdrant can return many highly similar chunks from the
        # same document. We keep the strongest chunks from each
        # document so that relevant evidence from other documents
        # has a chance to reach the reranker.
        # ---------------------------------------------------------
        candidates = _diversify_candidates(
            retrieved,
            max_per_document=4,
        )

        # ---------------------------------------------------------
        # 3. Rerank diversified candidates with the cross-encoder.
        # ---------------------------------------------------------
        reranked = self._reranker.rerank(
            query,
            candidates,
            top_n=self._rerank_top_n,
        )

        # ---------------------------------------------------------
        # 4. Select the strongest evidence.
        # ---------------------------------------------------------
        evidence = reranked[: self._evidence_top_n]

        if not evidence:
            return CitedAnswer(
                query=query,
                answer_text=_no_evidence_message(language),
                language=language,
                citations=[],
                retrieved_chunk_ids=[
                    chunk.chunk_id for chunk in retrieved
                ],
                legal_excerpts=[],
            )

        # ---------------------------------------------------------
        # 5. Build grounded prompt.
        # ---------------------------------------------------------
        prompt, citations = build_grounded_prompt(
            query,
            evidence,
            language=language,
        )

        # ---------------------------------------------------------
        # 6. Generate answer with the LLM.
        # ---------------------------------------------------------
        generated_answer = self._generator.generate(prompt).strip()

        # ---------------------------------------------------------
        # 7. Build original legal evidence.
        #
        # The LLM does NOT control:
        # - source file
        # - page
        # - section
        # - original legal text
        #
        # These values always come directly from the chunks.
        # ---------------------------------------------------------
        legal_excerpts = [
            LegalExcerpt(
                marker=citation.marker,
                text=chunk.text,
                source_file=chunk.source_file,
                section_title=chunk.section_title,
                page=chunk.page,
                chunk_id=chunk.chunk_id,
            )
            for citation, chunk in zip(
                citations,
                evidence,
                strict=True,
            )
        ]

        # ---------------------------------------------------------
        # 8. Validate numeric claims.
        # ---------------------------------------------------------
        evidence_text = "\n\n".join(
            chunk.text for chunk in evidence
        )

        is_valid, unsupported_numbers, _ = validate_numeric_claims(
            query,
            generated_answer,
            evidence_text,
        )

        # ---------------------------------------------------------
        # 9. Reject unsupported numeric claims.
        # ---------------------------------------------------------
        if not is_valid:
            answer_text = _validation_failure_message(
                language=language,
                unsupported_numbers=unsupported_numbers,
            )
        else:
            # -----------------------------------------------------
            # 10. Citation markers are added by Python, not LLM.
            # -----------------------------------------------------
            answer_text = _attach_citation(
                generated_answer,
                citations,
            )

        return CitedAnswer(
            query=query,
            answer_text=answer_text,
            language=language,
            citations=citations,
            retrieved_chunk_ids=[
                chunk.chunk_id
                for chunk in retrieved
            ],
            legal_excerpts=legal_excerpts,
        )


def _diversify_candidates(
    chunks: list[RetrievedChunk],
    max_per_document: int = 4,
) -> list[RetrievedChunk]:
    """Keep strong retrieval results while preventing one document
    from monopolizing the reranking stage.

    Qdrant results are already sorted by similarity score.

    The selection strategy is intentionally deterministic:

    1. Walk through the ranked Qdrant results.
    2. Keep at most `max_per_document` chunks from each document.
    3. Preserve the original Qdrant ordering.
    4. Do not change chunk scores or metadata.

    This is only candidate diversification. It does NOT decide
    which source is legally authoritative.
    """

    if not chunks:
        return []

    selected: list[RetrievedChunk] = []
    document_counts: dict[str, int] = defaultdict(int)

    for chunk in chunks:
        document_key = (
            chunk.document_id
            or chunk.source_file
            or chunk.chunk_id
        )

        if document_counts[document_key] >= max_per_document:
            continue

        selected.append(chunk)
        document_counts[document_key] += 1

    return selected


def _attach_citation(
    answer: str,
    citations,
) -> str:
    """Attach the strongest available citation to the answer.

    Citation metadata comes from Python rather than the LLM.
    """

    if not answer:
        return answer

    if not citations:
        return answer

    # The first citation corresponds to the highest-ranked evidence.
    marker = citations[0].marker

    # Avoid duplicating a citation if the model somehow generated one.
    if marker in answer:
        return answer

    return f"{answer} {marker}"


def _no_evidence_message(language: str) -> str:
    if language == "ar":
        return (
            "لم يتم العثور على مقاطع قانونية ذات صلة "
            "بالسؤال في قاعدة البيانات."
        )

    return (
        "No relevant legal excerpts were found in the indexed "
        "documents for this question."
    )


def _validation_failure_message(
    language: str,
    unsupported_numbers: set[str],
) -> str:
    if language == "ar":
        numbers = ", ".join(
            sorted(unsupported_numbers)
        )

        return (
            "تعذر اعتماد الإجابة المولدة تلقائيًا لأن "
            "بعض الأرقام الواردة فيها لا تتطابق مع النصوص "
            "القانونية المرجعية. "
            f"الأرقام غير المدعومة: {numbers}. "
            "يرجى الاعتماد على النص القانوني الأصلي "
            "المرفق أدناه."
        )

    numbers = ", ".join(
        sorted(unsupported_numbers)
    )

    return (
        "The generated answer could not be validated because "
        "some numbers in the answer are not supported by the "
        "retrieved legal evidence. "
        f"Unsupported numbers: {numbers}. "
        "Please rely on the original legal evidence below."
    )