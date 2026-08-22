"""End-to-end, fail-closed grounded legal RAG answer pipeline."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import replace
from typing import Protocol

from pydantic import ValidationError

from legal_rag.query.answer_language import answer_matches_language
from legal_rag.query.answer_validator import validate_numeric_claims
from legal_rag.query.evidence_sufficiency import (
    EvidenceAssessment,
    EvidenceSufficiencyEvaluator,
)
from legal_rag.query.models import (
    Citation,
    CitedAnswer,
    LegalExcerpt,
    RerankedChunk,
    RetrievalDiagnostics,
    RetrievedChunk,
)
from legal_rag.query.ollama_client import OllamaGenerationClient
from legal_rag.query.prompt_builder import GroundedPrompt, build_grounded_messages
from legal_rag.query.reranker import get_default_reranker
from legal_rag.query.retriever import RetrievalFilters
from legal_rag.query.structured_answer import GeneratedAnswer

_MODEL_CITATION_PATTERN = re.compile(r"\[\s*\d+\s*\]|\bE\d+\b", re.IGNORECASE)


class GenerationClient(Protocol):
    """Generation interface required by the answer pipeline."""

    def generate(
        self,
        prompt: str,
        temperature: float = 0.1,
        *,
        system: str | None = None,
        format_schema: Mapping[str, object] | None = None,
    ) -> str:
        """Return model-generated text."""
        ...


class RetrievalClient(Protocol):
    def search(
        self,
        query: str,
        top_k: int = 20,
        filters: RetrievalFilters | None = None,
        score_threshold: float | None = None,
    ) -> list[RetrievedChunk]:
        """Return dense retrieval candidates."""
        ...


class RerankingClient(Protocol):
    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_n: int = 6,
    ) -> list[RerankedChunk]:
        """Return cross-encoder-ranked candidates."""
        ...


class RAGAnswerPipeline:
    def __init__(
        self,
        retriever: RetrievalClient,
        reranker: RerankingClient | None = None,
        generator: GenerationClient | None = None,
        sufficiency_evaluator: EvidenceSufficiencyEvaluator | None = None,
        retrieve_top_k: int = 20,
        rerank_top_n: int = 6,
        evidence_top_n: int = 4,
        generation_temperature: float = 0.1,
        generation_retry_count: int = 1,
        maximum_context_characters: int = 12_000,
        maximum_dense_score_drop: float = 0.02,
    ) -> None:
        self._retriever = retriever
        self._reranker = reranker or get_default_reranker()
        self._generator = generator or OllamaGenerationClient()
        self._sufficiency_evaluator = sufficiency_evaluator or EvidenceSufficiencyEvaluator()
        self._retrieve_top_k = retrieve_top_k
        self._rerank_top_n = rerank_top_n
        self._evidence_top_n = evidence_top_n
        self._generation_temperature = generation_temperature
        self._generation_retry_count = generation_retry_count
        self._maximum_context_characters = maximum_context_characters
        self._maximum_dense_score_drop = maximum_dense_score_drop

    def answer(
        self,
        query: str,
        language: str = "mixed",
        filters: RetrievalFilters | None = None,
    ) -> CitedAnswer:
        retrieved = self._retriever.search(
            query,
            top_k=self._retrieve_top_k,
            filters=filters,
        )
        candidates = _diversify_candidates(retrieved, max_per_document=4)
        reranked = self._reranker.rerank(
            query,
            candidates,
            top_n=self._rerank_top_n,
        )
        assessment = self._sufficiency_evaluator.assess(query, retrieved, reranked)

        if not assessment.sufficient:
            return self._insufficient_answer(
                query=query,
                language=language,
                retrieved=retrieved,
                assessment=assessment,
            )

        evidence = _select_evidence(
            retrieved=retrieved,
            reranked=reranked,
            top_n=self._evidence_top_n,
            maximum_dense_score_drop=self._maximum_dense_score_drop,
        )
        prompt = build_grounded_messages(
            query,
            evidence,
            language=language,
            maximum_context_characters=self._maximum_context_characters,
        )
        generated = self._generate_structured(prompt, language=language)

        if generated is None:
            return self._generation_failure_answer(
                query=query,
                language=language,
                retrieved=retrieved,
                assessment=assessment,
                used_chunk_count=len(evidence),
                prompt_version=prompt.prompt_version,
            )

        if generated.insufficient_evidence:
            return self._insufficient_answer(
                query=query,
                language=language,
                retrieved=retrieved,
                assessment=assessment,
                reason="model_reported_insufficient_evidence",
                prompt_version=prompt.prompt_version,
            )

        selected_pairs = [
            (
                prompt.citations_by_evidence_id[evidence_id],
                prompt.chunks_by_evidence_id[evidence_id],
            )
            for evidence_id in generated.evidence_ids
        ]
        citations = [
            replace(citation, marker=f"[{index}]")
            for index, (citation, _) in enumerate(selected_pairs, start=1)
        ]
        selected_chunks = [chunk for _, chunk in selected_pairs]
        evidence_text = "\n\n".join(chunk.text for chunk in selected_chunks)
        is_valid, unsupported_numbers, _ = validate_numeric_claims(
            query,
            generated.answer,
            evidence_text,
        )

        if not is_valid:
            answer_text = _validation_failure_message(language, unsupported_numbers)
            citations = []
            selected_chunks = []
            assessment = replace(
                assessment,
                sufficient=False,
                reason="numeric_validation_failure",
            )
        else:
            answer_text = _attach_citations(generated.answer, citations)

        legal_excerpts = [
            LegalExcerpt(
                marker=citation.marker,
                text=chunk.text,
                source_file=chunk.source_file,
                section_title=chunk.section_title,
                page=chunk.page,
                chunk_id=chunk.chunk_id,
            )
            for citation, chunk in zip(citations, selected_chunks, strict=True)
        ]

        return CitedAnswer(
            query=query,
            answer_text=answer_text,
            language=language,
            citations=citations,
            retrieved_chunk_ids=[chunk.chunk_id for chunk in retrieved],
            legal_excerpts=legal_excerpts,
            retrieval=_diagnostics(
                assessment,
                candidate_count=len(retrieved),
                used_chunk_count=len(selected_chunks),
            ),
            prompt_version=prompt.prompt_version,
        )

    def _generate_structured(
        self,
        prompt: GroundedPrompt,
        *,
        language: str,
    ) -> GeneratedAnswer | None:
        schema = GeneratedAnswer.model_json_schema()
        attempts = self._generation_retry_count + 1

        for attempt in range(attempts):
            repair_instruction = ""
            if attempt:
                repair_instruction = _repair_instruction(language)

            raw_response = self._generator.generate(
                prompt.user + repair_instruction,
                temperature=self._generation_temperature,
                system=prompt.system,
                format_schema=schema,
            )
            try:
                generated = GeneratedAnswer.model_validate_json(raw_response)
            except ValidationError:
                continue

            allowed_ids = set(prompt.citations_by_evidence_id)
            returned_ids = set(generated.evidence_ids)
            if not returned_ids <= allowed_ids:
                continue
            if not generated.insufficient_evidence and not returned_ids:
                continue
            if _MODEL_CITATION_PATTERN.search(generated.answer):
                continue
            if not answer_matches_language(generated.answer, language):
                continue
            return generated

        return None

    def _insufficient_answer(
        self,
        *,
        query: str,
        language: str,
        retrieved: list[RetrievedChunk],
        assessment: EvidenceAssessment,
        reason: str | None = None,
        prompt_version: str | None = None,
    ) -> CitedAnswer:
        final_assessment = (
            assessment
            if reason is None
            else EvidenceAssessment(
                sufficient=False,
                reason=reason,
                top_dense_score=assessment.top_dense_score,
                dense_score_margin=assessment.dense_score_margin,
                top_rerank_score=assessment.top_rerank_score,
                exact_identifier_match=assessment.exact_identifier_match,
                source_count=assessment.source_count,
            )
        )
        return CitedAnswer(
            query=query,
            answer_text=_no_evidence_message(language),
            language=language,
            citations=[],
            retrieved_chunk_ids=[chunk.chunk_id for chunk in retrieved],
            legal_excerpts=[],
            retrieval=_diagnostics(
                final_assessment,
                candidate_count=len(retrieved),
                used_chunk_count=0,
            ),
            prompt_version=prompt_version,
        )

    def _generation_failure_answer(
        self,
        *,
        query: str,
        language: str,
        retrieved: list[RetrievedChunk],
        assessment: EvidenceAssessment,
        used_chunk_count: int,
        prompt_version: str,
    ) -> CitedAnswer:
        failed_assessment = EvidenceAssessment(
            sufficient=False,
            reason="invalid_structured_generation",
            top_dense_score=assessment.top_dense_score,
            dense_score_margin=assessment.dense_score_margin,
            top_rerank_score=assessment.top_rerank_score,
            exact_identifier_match=assessment.exact_identifier_match,
            source_count=assessment.source_count,
        )
        return CitedAnswer(
            query=query,
            answer_text=_generation_failure_message(language),
            language=language,
            citations=[],
            retrieved_chunk_ids=[chunk.chunk_id for chunk in retrieved],
            legal_excerpts=[],
            retrieval=_diagnostics(
                failed_assessment,
                candidate_count=len(retrieved),
                used_chunk_count=used_chunk_count,
            ),
            prompt_version=prompt_version,
        )


def _diversify_candidates(
    chunks: list[RetrievedChunk],
    max_per_document: int = 4,
) -> list[RetrievedChunk]:
    """Keep at most ``max_per_document`` candidates from each document."""

    selected: list[RetrievedChunk] = []
    document_counts: dict[str, int] = defaultdict(int)
    for chunk in chunks:
        document_key = chunk.document_id or chunk.source_file or chunk.chunk_id
        if document_counts[document_key] >= max_per_document:
            continue
        selected.append(chunk)
        document_counts[document_key] += 1
    return selected


def _select_evidence(
    *,
    retrieved: list[RetrievedChunk],
    reranked: list[RerankedChunk],
    top_n: int,
    maximum_dense_score_drop: float,
) -> list[RerankedChunk]:
    """Preserve dense top and admit only similarly strong extra chunks."""

    if top_n <= 0 or not retrieved:
        return []

    reranked_by_id = {chunk.chunk_id: chunk for chunk in reranked}
    dense_top = retrieved[0]
    selected = [reranked_by_id.get(dense_top.chunk_id) or _as_reranked(dense_top)]
    selected_ids = {dense_top.chunk_id}

    for chunk in reranked:
        if len(selected) >= top_n:
            break
        if chunk.chunk_id in selected_ids:
            continue
        if dense_top.score - chunk.score > maximum_dense_score_drop:
            continue
        selected.append(chunk)
        selected_ids.add(chunk.chunk_id)
    return selected


def _as_reranked(chunk: RetrievedChunk) -> RerankedChunk:
    return RerankedChunk(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        score=chunk.score,
        text=chunk.text,
        source_file=chunk.source_file,
        section_type=chunk.section_type,
        section_title=chunk.section_title,
        page=chunk.page,
        language=chunk.language,
        document_type=chunk.document_type,
        source=chunk.source,
        payload=chunk.payload,
        rerank_score=float("-inf"),
    )


def _diagnostics(
    assessment: EvidenceAssessment,
    *,
    candidate_count: int,
    used_chunk_count: int,
) -> RetrievalDiagnostics:
    return RetrievalDiagnostics(
        strategy="dense_plus_cross_encoder",
        candidate_count=candidate_count,
        used_chunk_count=used_chunk_count,
        sufficient=assessment.sufficient,
        reason=assessment.reason,
        top_dense_score=assessment.top_dense_score,
        dense_score_margin=assessment.dense_score_margin,
        top_rerank_score=assessment.top_rerank_score,
        exact_identifier_match=assessment.exact_identifier_match,
        source_count=assessment.source_count,
    )


def _attach_citations(answer: str, citations: list[Citation]) -> str:
    markers = " ".join(citation.marker for citation in citations)
    return f"{answer.strip()} {markers}".strip()


def _repair_instruction(language: str) -> str:
    if language == "ar":
        return (
            "\n\nكانت الاستجابة السابقة غير صالحة. أعد كائن JSON مطابقاً للمخطط فقط، "
            "واكتب حقل answer باللغة العربية فقط، واستخدم حصراً evidence_ids المتاحة."
        )
    return (
        "\n\nThe previous response was invalid. Return schema-valid JSON only, "
        "write the answer in the required language, and use only supplied evidence_ids."
    )


def _no_evidence_message(language: str) -> str:
    if language == "ar":
        return "المعلومات المتاحة في المستندات المفهرسة غير كافية للإجابة عن هذا السؤال."
    return "The indexed documents do not contain sufficient evidence to answer this question."


def _generation_failure_message(language: str) -> str:
    if language == "ar":
        return "تعذر إنتاج إجابة يمكن التحقق من استشهاداتها من الأدلة المتاحة."
    return "A citation-valid answer could not be produced from the available evidence."


def _validation_failure_message(language: str, unsupported_numbers: set[str]) -> str:
    numbers = ", ".join(sorted(unsupported_numbers))
    if language == "ar":
        return f"تعذر اعتماد الإجابة لأن أرقاماً غير مدعومة ظهرت فيها: {numbers}."
    return f"The answer contained unsupported numeric values: {numbers}."
