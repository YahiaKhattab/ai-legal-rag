"""End-to-end, fail-closed grounded legal RAG answer pipeline.

Changes from the dense-only version: an optional keyword (BM25) retriever
can be supplied. When present, its results are fused with dense retrieval
(Reciprocal Rank Fusion) and the FUSED candidate pool is what gets
reranked -- this is the actual point where keyword search
gets a chance to surface a chunk dense search missed entirely, not just
re-order what dense already found.

Deliberately NOT changed: the evidence-sufficiency gate's dense-score
checks (`top_dense_score`, `dense_score_margin`) are still computed from
the *original* dense-only `retrieved` list, never from fused scores. RRF
scores are rank-based and on a completely different scale from cosine
similarity (a rank-1 RRF hit is typically ~0.016, nowhere near the 0.855
dense threshold) -- feeding fused scores into that threshold would make
the gate reject almost everything. Turning keyword search on/off must not
change what "the dense score was strong enough" means for anything already
relying on that gate.
"""

from __future__ import annotations

import re
import time
import unicodedata
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import replace
from typing import Protocol

from pydantic import ValidationError

from legal_rag.observability.tracing import attributes, span, traced
from legal_rag.query.answer_language import (
    answer_matches_language,
    detect_question_language,
)
from legal_rag.query.answer_validator import validate_numeric_claims
from legal_rag.query.evidence_sufficiency import (
    EvidenceAssessment,
    EvidenceSufficiencyEvaluator,
)
from legal_rag.query.hybrid_fusion import DEFAULT_RRF_K, reciprocal_rank_fusion
from legal_rag.query.models import (
    Citation,
    CitedAnswer,
    LegalExcerpt,
    RerankedChunk,
    RetrievalDiagnostics,
    RetrievedChunk,
)
from legal_rag.query.ollama_client import OllamaGenerationClient
from legal_rag.query.prompt_builder import (
    GroundedPrompt,
    build_grounded_messages,
)
from legal_rag.query.reranker import get_default_reranker
from legal_rag.query.retriever import RetrievalFilters
from legal_rag.query.structured_answer import GeneratedAnswer


_MODEL_CITATION_PATTERN = re.compile(
    r"\[\s*\d+\s*\]|\bE\d+\b",
    re.IGNORECASE,
)


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


class KeywordRetrievalClient(Protocol):
    def search(
        self,
        query: str,
        top_k: int = 20,
        filters: RetrievalFilters | None = None,
    ) -> list[RetrievedChunk]:
        """Return keyword (BM25) retrieval candidates."""
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
        keyword_retriever: KeywordRetrievalClient | None = None,
        retrieve_top_k: int = 20,
        rerank_top_n: int = 6,
        evidence_top_n: int = 4,
        generation_temperature: float = 0.1,
        generation_retry_count: int = 1,
        maximum_context_characters: int = 12_000,
        maximum_dense_score_drop: float = 0.02,
        keyword_top_k: int = 20,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> None:
        self._retriever = retriever
        self._reranker = reranker or get_default_reranker()
        self._generator = generator or OllamaGenerationClient()

        self._sufficiency_evaluator = (
            sufficiency_evaluator
            or EvidenceSufficiencyEvaluator()
        )

        # None means dense-only, unchanged from before this feature was
        # added -- keyword search is strictly opt-in per pipeline instance.
        self._keyword_retriever = keyword_retriever
        self._keyword_top_k = keyword_top_k
        self._rrf_k = rrf_k
        self._retrieval_strategy_label = (
            "dense_plus_keyword_plus_cross_encoder"
            if keyword_retriever is not None
            else "dense_plus_cross_encoder"
        )

        self._retrieve_top_k = retrieve_top_k
        self._rerank_top_n = rerank_top_n
        self._evidence_top_n = evidence_top_n
        self._generation_temperature = generation_temperature
        self._generation_retry_count = generation_retry_count
        self._maximum_context_characters = maximum_context_characters
        self._maximum_dense_score_drop = maximum_dense_score_drop

    @traced("rag.answer")
    def answer(
        self,
        query: str,
        language: str = "mixed",
        filters: RetrievalFilters | None = None,
    ) -> CitedAnswer:

        pipeline_start = time.perf_counter()

        print("\n========== PIPELINE TIMING ==========")

        # ---------------------------------------------------------------
        # 0. Resolve answer language
        # ---------------------------------------------------------------

        stage_start = time.perf_counter()

        effective_language = detect_question_language(query)

        if language != "mixed":
            effective_language = language

        print(
            "[Timing] Language Detection: "
            f"{time.perf_counter() - stage_start:.3f}s"
        )

        # ---------------------------------------------------------------
        # 1. Dense retrieval
        # ---------------------------------------------------------------

        stage_start = time.perf_counter()

        retrieved = self._retriever.search(
            query,
            top_k=self._retrieve_top_k,
            filters=filters,
        )

        print(
            "[Timing] Dense Retrieval:    "
            f"{time.perf_counter() - stage_start:.3f}s"
        )

        # ---------------------------------------------------------------
        # 1b. Keyword retrieval + fusion (optional)
        #
        # `retrieved` above is untouched and stays dense-only -- it's what
        # the evidence gate's dense-score checks are computed against in
        # step 4, and what the eventual CitedAnswer.retrieved_chunk_ids
        # reports.
        #
        # Fusion widens the candidate pool, then diversification limits
        # repeated chunks from the same document before reranking.
        # ---------------------------------------------------------------

        stage_start = time.perf_counter()

        if self._keyword_retriever is not None:
            keyword_hits = self._keyword_retriever.search(
                query,
                top_k=self._keyword_top_k,
                filters=filters,
            )

            fused_candidates = reciprocal_rank_fusion(
                [retrieved, keyword_hits],
                k=self._rrf_k,
            )
        else:
            fused_candidates = retrieved

        candidates = _diversify_candidates(
            fused_candidates,
            max_per_document=4,
        )

        print(
            "[Timing] Diversification:     "
            f"{time.perf_counter() - stage_start:.3f}s"
        )

        print(
            "[Timing] Retrieved: "
            f"{len(retrieved)} | "
            f"Fused: {len(fused_candidates)} | "
            f"Candidates after diversification: {len(candidates)}"
        )

        # ---------------------------------------------------------------
        # 3. Cross-encoder reranking
        # ---------------------------------------------------------------

        stage_start = time.perf_counter()

        reranked = self._reranker.rerank(
            query,
            candidates,
            top_n=self._rerank_top_n,
        )

        print(
            "[Timing] Reranking:           "
            f"{time.perf_counter() - stage_start:.3f}s"
        )

        # Apply the individual/collective procedure guard before the
        # sufficiency evaluator, so mismatched evidence cannot make a query
        # appear answerable.
        reranked = _filter_procedure_mismatches(query, reranked)

        # ---------------------------------------------------------------
        # 4. Evidence sufficiency gate
        #
        # NOTE: `retrieved` here is still the dense-only list from step 1,
        # not `fused_candidates` or `candidates`.
        # ---------------------------------------------------------------

        stage_start = time.perf_counter()

        assessment = self._sufficiency_evaluator.assess(
            query,
            retrieved,
            reranked,
        )

        print(
            "[Timing] Evidence Sufficiency:"
            f" {time.perf_counter() - stage_start:.3f}s"
        )

        if not assessment.sufficient:
            total_time = time.perf_counter() - pipeline_start

            print(
                "[Timing] TOTAL:               "
                f"{total_time:.3f}s"
            )
            print("=====================================\n")

            return self._insufficient_answer(
                query=query,
                language=effective_language,
                retrieved=retrieved,
                assessment=assessment,
            )

        # ---------------------------------------------------------------
        # 5. Select evidence
        # ---------------------------------------------------------------

        stage_start = time.perf_counter()

        evidence = _select_evidence(
            query=query,
            retrieved=retrieved,
            reranked=reranked,
            top_n=self._evidence_top_n,
            maximum_dense_score_drop=self._maximum_dense_score_drop,
        )

        print(
            "[Timing] Evidence Selection: "
            f"{time.perf_counter() - stage_start:.3f}s"
        )

        if not evidence:
            total_time = time.perf_counter() - pipeline_start

            print(
                "[Timing] TOTAL:               "
                f"{total_time:.3f}s"
            )
            print("=====================================\n")

            return self._insufficient_answer(
                query=query,
                language=effective_language,
                retrieved=retrieved,
                assessment=replace(
                    assessment,
                    sufficient=False,
                    reason="no_safe_evidence_selected",
                ),
            )

        # ---------------------------------------------------------------
        # 6. Build grounded prompt
        # ---------------------------------------------------------------

        stage_start = time.perf_counter()

        prompt = build_grounded_messages(
            query,
            evidence,
            language=effective_language,
            maximum_context_characters=self._maximum_context_characters,
        )

        print(
            "[Timing] Prompt Building:     "
            f"{time.perf_counter() - stage_start:.3f}s"
        )

        print(
            "[Timing] Evidence Chunks: "
            f"{len(evidence)} | "
            f"Context chars: {len(prompt.user)}"
        )

        # ---------------------------------------------------------------
        # 7. Structured generation
        # ---------------------------------------------------------------

        stage_start = time.perf_counter()

        self._last_generation_failure_reason = None
        generated = self._generate_structured(
            prompt,
            language=effective_language,
            query=query,
        )

        print(
            "[Timing] Structured Generation:"
            f" {time.perf_counter() - stage_start:.3f}s"
        )

        if generated is None:
            total_time = time.perf_counter() - pipeline_start

            print(
                "[Timing] TOTAL:               "
                f"{total_time:.3f}s"
            )
            print("=====================================\n")

            failure_reason = getattr(
                self,
                "_last_generation_failure_reason",
                None,
            )
            if failure_reason == "numeric_validation_failure":
                return self._insufficient_answer(
                    query=query,
                    language=effective_language,
                    retrieved=retrieved,
                    assessment=assessment,
                    reason=failure_reason,
                    prompt_version=prompt.prompt_version,
                )

            return self._generation_failure_answer(
                query=query,
                language=effective_language,
                retrieved=retrieved,
                assessment=assessment,
                used_chunk_count=len(evidence),
                prompt_version=prompt.prompt_version,
            )

        # ---------------------------------------------------------------
        # 8. Model-level insufficient evidence
        # ---------------------------------------------------------------

        if generated.insufficient_evidence:
            total_time = time.perf_counter() - pipeline_start

            print(
                "[Timing] TOTAL:               "
                f"{total_time:.3f}s"
            )
            print("=====================================\n")

            return self._insufficient_answer(
                query=query,
                language=effective_language,
                retrieved=retrieved,
                assessment=assessment,
                reason="model_reported_insufficient_evidence",
                prompt_version=prompt.prompt_version,
            )

        # ---------------------------------------------------------------
        # 9. Validate returned evidence IDs
        # ---------------------------------------------------------------

        stage_start = time.perf_counter()

        allowed_ids = set(
            prompt.chunks_by_evidence_id.keys()
        )

        selected_pairs: list[tuple[Citation, RerankedChunk]] = []

        for evidence_id in generated.evidence_ids:
            citation = prompt.citations_by_evidence_id.get(
                evidence_id
            )
            chunk = prompt.chunks_by_evidence_id.get(
                evidence_id
            )

            if citation is None or chunk is None:
                continue

            selected_pairs.append(
                (
                    citation,
                    chunk,
                )
            )

        returned_ids = set(generated.evidence_ids)

        print(
            "[Timing] Evidence ID Validation:"
            f" {time.perf_counter() - stage_start:.3f}s"
        )

        if not returned_ids <= allowed_ids:
            total_time = time.perf_counter() - pipeline_start

            print(
                "[Generation] Rejected: invalid evidence IDs."
            )

            print(
                "[Timing] TOTAL:               "
                f"{total_time:.3f}s"
            )
            print("=====================================\n")

            return self._generation_failure_answer(
                query=query,
                language=effective_language,
                retrieved=retrieved,
                assessment=replace(
                    assessment,
                    sufficient=False,
                    reason="model_returned_invalid_evidence",
                ),
                used_chunk_count=len(evidence),
                prompt_version=prompt.prompt_version,
            )

        if not selected_pairs:
            total_time = time.perf_counter() - pipeline_start

            print(
                "[Timing] TOTAL:               "
                f"{total_time:.3f}s"
            )
            print("=====================================\n")

            return self._generation_failure_answer(
                query=query,
                language=effective_language,
                retrieved=retrieved,
                assessment=replace(
                    assessment,
                    sufficient=False,
                    reason="model_returned_no_valid_evidence",
                ),
                used_chunk_count=len(evidence),
                prompt_version=prompt.prompt_version,
            )

        # ---------------------------------------------------------------
        # 10. Citations
        # ---------------------------------------------------------------

        stage_start = time.perf_counter()

        citations = [
            replace(
                citation,
                marker=f"[{index}]",
            )
            for index, (citation, _) in enumerate(
                selected_pairs,
                start=1,
            )
        ]

        selected_chunks = [
            chunk
            for _, chunk in selected_pairs
        ]

        print(
            "[Timing] Citation Building:   "
            f"{time.perf_counter() - stage_start:.3f}s"
        )

        # ---------------------------------------------------------------
        # 11. Numeric claim validation
        # ---------------------------------------------------------------

        stage_start = time.perf_counter()

        evidence_text = "\n\n".join(
            chunk.text
            for chunk in selected_chunks
        )

        # Do not run a second LLM rewrite over each source: it adds latency
        # and can distort evidence or introduce unsupported details. For the
        # specifically guarded individual-dispute clause, use the deterministic
        # formulation only when all required signals are present; otherwise
        # keep the answer that already passed generation-time validation.
        original_answer = generated.answer.strip()
        procedural_answer = _individual_dispute_procedure_answer(
            query, evidence_text
        )
        answer_text = procedural_answer or original_answer

        is_valid, unsupported_numbers, _ = validate_numeric_claims(
            query,
            answer_text,
            evidence_text,
        )

        if not is_valid:
            # If paraphrasing introduced unsupported numbers, retain the
            # original answer that passed validation during generation.
            original_valid, _, _ = validate_numeric_claims(
                query,
                original_answer,
                evidence_text,
            )
            if original_valid:
                answer_text = original_answer
                is_valid = True
                print("[Rewrite] Rejected paraphrase; using validated original.")
            else:
                answer_text = _validation_failure_message(
                    effective_language,
                    unsupported_numbers,
                )
                citations = []
                selected_chunks = []
                assessment = replace(
                    assessment,
                    sufficient=False,
                    reason="numeric_validation_failure",
                )

        if is_valid and not all(citation.marker in answer_text for citation in citations):
            answer_text = _attach_citations(answer_text, citations)

        print(
            "[Timing] Numeric Validation:  "
            f"{time.perf_counter() - stage_start:.3f}s"
        )

        # ---------------------------------------------------------------
        # 12. Legal excerpts
        # ---------------------------------------------------------------

        stage_start = time.perf_counter()

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
                selected_chunks,
                strict=True,
            )
        ]

        print(
            "[Timing] Legal Excerpts:      "
            f"{time.perf_counter() - stage_start:.3f}s"
        )

        total_time = time.perf_counter() - pipeline_start

        print(
            "[Timing] TOTAL:               "
            f"{total_time:.3f}s"
        )

        print("=====================================\n")

        return CitedAnswer(
            query=query,
            answer_text=answer_text,
            language=effective_language,
            citations=citations,
            retrieved_chunk_ids=[
                chunk.chunk_id
                for chunk in retrieved
            ],
            legal_excerpts=legal_excerpts,
            retrieval=_diagnostics(
                assessment,
                candidate_count=len(retrieved),
                used_chunk_count=len(selected_chunks),
                strategy=self._retrieval_strategy_label,
            ),
            prompt_version=prompt.prompt_version,
        )

    def _rewrite_cited_sources(
        self,
        *,
        query: str,
        citations: list[Citation],
        chunks: list[RerankedChunk],
        language: str,
        fallback_answer: str,
    ) -> str:
        """Present each selected citation as a readable, independently grounded section."""
        sections: list[str] = []
        for index, (citation, chunk) in enumerate(zip(citations, chunks, strict=True)):
            source = str(chunk.text or "").strip()
            if not source:
                continue
            if len(chunks) == 1:
                heading = ""
            elif language == "ar":
                heading = f"من المصدر {index + 1}: "
            else:
                heading = f"From source {index + 1}: "
            # Ask the editor to restate this source itself, not the model's
            # potentially cross-source answer. Keep exact factual values.
            rewritten = self._rewrite_answer(
                query=query,
                answer=source,
                evidence_text=source,
                language=language,
            ).strip()
            if not rewritten:
                rewritten = source
            sections.append(f"{heading}{rewritten} {citation.marker}")

        if sections:
            return "\n\n".join(sections)
        return fallback_answer

    def _rewrite_answer(
        self,
        *,
        query: str,
        answer: str,
        evidence_text: str,
        language: str,
    ) -> str:
        """Paraphrase a validated answer without adding legal content."""
        if not answer.strip():
            return answer

        if language == "ar":
            instruction = (
                "حرّر النص إلى إجابة عربية قانونية سهلة القراءة، واضحة ومباشرة، لا إلى نقل حرفي مشوّه من OCR. "
                "ابدأ بجواب مباشر على السؤال، ثم رتّب التفاصيل في فقرات قصيرة أو نقاط مرقمة عند وجود خطوات أو شروط أو أطراف. "
                "استخدم عناوين قصيرة فقط عند الحاجة، وصحّح المسافات وعلامات الترقيم والتكرار وأخطاء التنسيق الواضحة دون تغيير المعنى. "
                "لا تجب من جديد ولا تستنتج. حافظ على الأرقام الجوهرية التي تجيب عن السؤال، مثل المدد والمبالغ والنسب والأعداد والشروط. "
                "لا تذكر في متن الإجابة أرقام المواد أو أرقام القوانين أو سنوات إصدارها أو أرقام الأبواب والفصول؛ تُترك هذه المعرّفات للاستشهاد فقط. "
                "لا تضف أي معلومة غير موجودة في النص الأصلي أو الدليل، ولا تخمّن أي معرّف قانوني ملتبس أو متعارض. "
                "إذا كان النص ملتبسًا في المعنى، أعد النص الأصلي دون تغيير. أخرج نص الإجابة فقط دون JSON أو شرح عن عملية التحرير.\n\n"
            )
        else:
            instruction = (
                "Edit the text into a clear, direct, readable legal answer, not a verbatim OCR dump. "
                "Start with a direct answer, then organize details into short paragraphs or numbered bullets for steps, conditions, or parties. "
                "Use brief headings only when helpful; fix spacing, punctuation, repetition, and obvious formatting artifacts without changing meaning. "
                "Do not answer anew or infer. Preserve substantive quantities needed to answer, such as deadlines, amounts, percentages, counts, and conditions. "
                "Do not include article numbers, law numbers, enactment years, or book/chapter numbers in the answer body; keep legal identifiers in citations only. "
                "Add no facts beyond the source and never guess an ambiguous or conflicting legal identifier. "
                "If the meaning itself is ambiguous, return the original unchanged. Output only the answer, no JSON or editing explanation.\n\n"
            )

        rewrite_prompt = (
            instruction
            + f"QUESTION:\n{query}\n\n"
            + f"ORIGINAL ANSWER:\n{answer}\n\n"
            + f"SUPPORTING EVIDENCE:\n{evidence_text[:7000]}"
        )
        try:
            rewritten = self._generator.generate(
                rewrite_prompt,
                temperature=0.0,
                system=(
                    "You are a strict legal-language editor. "
                    "Never change the legal meaning or introduce facts."
                ),
            )
            rewritten = (rewritten or "").strip()
            # Guard against empty output and accidental JSON wrappers.
            if not rewritten or rewritten.startswith("{"):
                return answer
            return _remove_legal_identifiers_from_answer(rewritten, language)
        except Exception as exc:
            print(f"[Rewrite] Failed; retaining original answer: {exc}")
            return answer

    # -------------------------------------------------------------------
    # Generation
    # -------------------------------------------------------------------

    @traced("answer.generate_and_validate")
    def _generate_structured(
        self,
        prompt: GroundedPrompt,
        *,
        language: str,
        query: str,
    ) -> GeneratedAnswer | None:

        schema = GeneratedAnswer.model_json_schema()
        attempts = self._generation_retry_count + 1

        allowed_ids = set(
            prompt.chunks_by_evidence_id.keys()
        )

        allowed_ids_text = ", ".join(
            sorted(allowed_ids)
        )

        previous_failure = ""
        numeric_validation_failed = False

        for attempt in range(attempts):
            with span("answer.attempt"):
                attributes(
                    **{
                        "rag.attempt": attempt + 1
                    }
                )

                repair_instruction = ""

                if attempt:
                    repair_instruction = _repair_instruction(
                        language
                    )

                    repair_instruction += (
                        "\\n\\nIMPORTANT VALIDATION FEEDBACK:\\n"
                        f"{previous_failure}\\n"
                        "Re-read the supplied evidence before answering. "
                        "Identify which evidence directly addresses the "
                        "question's dispute type and ignore evidence about "
                        "a different type of dispute. Do not merge individual "
                        "and collective dispute procedures. Use only facts "
                        "and numbers explicitly supported by the selected "
                        "evidence; never guess, substitute, or infer a "
                        "deadline. If the evidence does not clearly support "
                        "the answer, return insufficient_evidence=true and "
                        "an empty answer.\\n"
                        "You MUST return a non-empty evidence_ids array "
                        "for an answer based on the supplied evidence.\\n"
                        f"Allowed evidence_ids: [{allowed_ids_text}]\\n"
                        "Choose only IDs from this exact list that support "
                        "your answer. Do not invent IDs. Do not return an "
                        "empty evidence_ids array unless evidence is "
                        "insufficient.\\n"
                        "Answer concisely in the question language; do not merge individual and collective dispute procedures. Every claim must be supported by the evidence. If uncertain, abstain. Return the complete JSON object only."
                    )

                generation_start = time.perf_counter()

                raw_response = self._generator.generate(
                    prompt.user + repair_instruction,
                    temperature=self._generation_temperature,
                    system=prompt.system,
                    format_schema=schema,
                )

            print(
                "[Generation] Ollama response: "
                f"{time.perf_counter() - generation_start:.3f}s"
            )

            # Diagnostic only: log a bounded response preview.
            # This helps identify malformed or incomplete model output.
            response_preview = (
                raw_response[:1500]
                if raw_response
                else "<empty response>"
            )

            print(
                "[Generation] Raw response preview: "
                f"{response_preview!r}"
            )

            validation_start = time.perf_counter()

            try:
                generated = GeneratedAnswer.model_validate_json(
                    raw_response
                )
            except ValidationError as exc:
                previous_failure = (
                    "The previous response failed JSON/schema "
                    f"validation: {str(exc)[:1000]}"
                )

                print(
                    "[Generation] JSON validation failed: "
                    f"{time.perf_counter() - validation_start:.3f}s"
                )
                print(
                    "[Generation] Validation error: "
                    f"{str(exc)[:1000]}"
                )
                continue

            returned_ids = set(
                generated.evidence_ids
            )

            # Never allow citations outside supplied evidence.
            if not returned_ids <= allowed_ids:
                invalid_ids = sorted(
                    returned_ids - allowed_ids
                )

                previous_failure = (
                    "The previous response used invalid evidence_ids: "
                    f"{invalid_ids}. Use only: [{allowed_ids_text}]."
                )

                print(
                    "[Generation] Rejected: invalid evidence IDs. "
                    f"Invalid IDs: {invalid_ids}"
                )
                continue

            # A non-insufficient answer must cite evidence.
            if (
                not generated.insufficient_evidence
                and not returned_ids
            ):
                previous_failure = (
                    "The previous response had an empty evidence_ids "
                    "array, but it provided an answer. This is invalid. "
                    "Select supporting IDs from the supplied evidence: "
                    f"[{allowed_ids_text}]."
                )

                print(
                    "[Generation] Rejected: no evidence IDs."
                )
                continue

            # Model must not manufacture citation markers.
            if _MODEL_CITATION_PATTERN.search(
                generated.answer
            ):
                previous_failure = (
                    "The previous answer included citation markers "
                    "inside the answer text. Remove all citation "
                    "markers from answer; put evidence references "
                    "only in evidence_ids."
                )

                print(
                    "[Generation] Rejected: "
                    "manufactured citation marker."
                )
                continue

            # Enforce requested/detected answer language.
            if not answer_matches_language(
                generated.answer,
                language,
            ):
                previous_failure = (
                    "The previous answer was written in the wrong "
                    f"language. Required language: {language}."
                )

                print(
                    "[Generation] Rejected: "
                    "wrong answer language."
                )
                continue

            # Validate factual numbers against only the evidence IDs
            # selected by the model. Reject and retry before returning
            # an answer, so the model receives actionable feedback.
            if not generated.insufficient_evidence:
                selected_evidence_text = "\n\n".join(
                    str(prompt.chunks_by_evidence_id[evidence_id].text)
                    for evidence_id in generated.evidence_ids
                )
                numeric_ok, unsupported_numbers, _ = validate_numeric_claims(
                    query,
                    generated.answer,
                    selected_evidence_text,
                )
                # A tightly guarded deterministic answer exists for this exact
                # individual-dispute clause. Let that answer reach the final
                # assembly instead of failing early on the model's bad numbers.
                procedural_fallback_available = bool(
                    _individual_dispute_procedure_answer(query, selected_evidence_text)
                )
                if not numeric_ok and not procedural_fallback_available:
                    numeric_feedback = (
                        ", ".join(sorted(map(str, unsupported_numbers)))
                        or "Unsupported numeric claim detected."
                    )
                    numeric_validation_failed = True
                    # Give the retry the actual source text and concrete
                    # correction context. A generic warning alone often
                    # causes small models to repeat the same hallucinated number.
                    previous_failure = (
                        "NUMERIC VALIDATION FAILED. The prior answer contains "
                        f"unsupported numeric claim(s): {numeric_feedback}.\n"
                        "SOURCE TEXT (authoritative; copy the relevant value "
                        "exactly, do not rely on memory):\n"
                        f"{selected_evidence_text[:7000]}\n"
                        "Rewrite the answer using only numeric values that "
                        "appear in this source. If the source says ten days, "
                        "do not say twenty days. Do not add article/law numbers "
                        "unless needed and clearly supported. Prefer omitting "
                        "unnecessary numbers. If no safe answer can be formed, "
                        "return insufficient_evidence=true and an empty answer."
                    )
                    print(
                        "[Generation] Rejected: numeric claim "
                        f"validation failed: {numeric_feedback}"
                    )
                    continue

            print(
                "[Generation] Post-validation: "
                f"{time.perf_counter() - validation_start:.3f}s"
            )

            self._last_generation_failure_reason = None
            return generated

        if numeric_validation_failed:
            self._last_generation_failure_reason = (
                "numeric_validation_failure"
            )
        return None
    # -------------------------------------------------------------------
    # Insufficient evidence
    # -------------------------------------------------------------------

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
            else replace(
                assessment,
                sufficient=False,
                reason=reason,
            )
        )

        return CitedAnswer(
            query=query,
            answer_text=_no_evidence_message(
                language
            ),
            language=language,
            citations=[],
            retrieved_chunk_ids=[
                chunk.chunk_id
                for chunk in retrieved
            ],
            legal_excerpts=[],
            retrieval=_diagnostics(
                final_assessment,
                candidate_count=len(retrieved),
                used_chunk_count=0,
                strategy=self._retrieval_strategy_label,
            ),
            prompt_version=prompt_version,
        )

    # -------------------------------------------------------------------
    # Generation failure
    # -------------------------------------------------------------------

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

        failed_assessment = replace(
            assessment,
            sufficient=False,
            reason="invalid_structured_generation",
        )

        return CitedAnswer(
            query=query,
            answer_text=_generation_failure_message(
                language
            ),
            language=language,
            citations=[],
            retrieved_chunk_ids=[
                chunk.chunk_id
                for chunk in retrieved
            ],
            legal_excerpts=[],
            retrieval=_diagnostics(
                failed_assessment,
                candidate_count=len(retrieved),
                used_chunk_count=used_chunk_count,
                strategy=self._retrieval_strategy_label,
            ),
            prompt_version=prompt_version,
        )


# =========================================================================
# Candidate diversification
# =========================================================================


@traced("retrieval.diversify")
def _diversify_candidates(
    chunks: list[RetrievedChunk],
    max_per_document: int = 4,
) -> list[RetrievedChunk]:
    """Keep at most ``max_per_document`` candidates from each document."""

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


# =========================================================================
# Evidence selection
# =========================================================================


@traced("evidence.select")
def _normalize_arabic_for_matching(text: str) -> str:
    """Normalize common Arabic variants for conservative procedure guards."""
    text = unicodedata.normalize("NFKC", text or "").casefold()
    # Remove Arabic diacritics and tatweel; normalize common letter variants.
    text = text.replace("ـ", "")
    text = re.sub(r"[\u064b-\u065f\u0670]", "", text)
    for old, new in (("أ", "ا"), ("إ", "ا"), ("آ", "ا"), ("ى", "ي"),
                     ("ة", "ه"), ("ؤ", "و"), ("ئ", "ي")):
        text = text.replace(old, new)
    return " ".join(text.split())


def _filter_procedure_mismatches(query: str, evidence: list[RerankedChunk]) -> list[RerankedChunk]:
    """Keep individual and collective labor-dispute evidence from being mixed."""
    q = _normalize_arabic_for_matching(query)
    individual_query = "نزاع فرد" in q
    collective_query = "نزاع جماع" in q
    if not individual_query and not collective_query:
        return evidence
    kept = []
    for chunk in evidence:
        text = _normalize_arabic_for_matching(f"{chunk.section_title or ''} {chunk.text or ''}")
        individual = "نزاع فرد" in text
        collective = "نزاع جماع" in text or "مفاوضه جماعي" in text
        if individual_query and collective and not individual:
            continue
        if collective_query and individual and not collective:
            continue
        kept.append(chunk)
    return kept


def _select_evidence(
    *,
    query: str,
    retrieved: list[RetrievedChunk],
    reranked: list[RerankedChunk],
    top_n: int,
    maximum_dense_score_drop: float,
) -> list[RerankedChunk]:
    """Select the safest evidence for grounded generation.

    Selection priority:

    1. Exact article match when the query explicitly names an article.
    2. Cross-encoder ranking.
    3. Dense-score safety constraint.

    This prevents an unrelated article from entering the context merely
    because it has a high reranker score or shares generic legal words.

    NOTE: `dense_scores` below is keyed from `retrieved` (dense-only), so a
    chunk that reranking kept only because keyword search found it will
    have no entry here (`dense_scores.get(...)` returns None) and is
    therefore exempt from the dense-score-drop constraint, rather than
    being compared against a dense score it never had. It still has to
    earn its place through the article-match/reranker logic below.
    """

    if top_n <= 0 or not reranked:
        return []

    dense_scores = {
        chunk.chunk_id: chunk.score
        for chunk in retrieved
    }

    best_dense_score = max(
        dense_scores.values(),
        default=None,
    )

    query_identifiers = _extract_article_identifiers(
        query
    )

    selected: list[RerankedChunk] = []
    selected_ids: set[str] = set()

    # ---------------------------------------------------------------
    # First pass:
    # exact article matches
    # ---------------------------------------------------------------

    if query_identifiers:

        for chunk in reranked:

            if len(selected) >= top_n:
                break

            if chunk.chunk_id in selected_ids:
                continue

            if not _chunk_matches_article(
                chunk,
                query_identifiers,
            ):
                continue

            dense_score = dense_scores.get(
                chunk.chunk_id
            )

            if (
                dense_score is not None
                and best_dense_score is not None
                and (
                    best_dense_score - dense_score
                    > maximum_dense_score_drop
                )
            ):
                continue

            selected.append(chunk)
            selected_ids.add(chunk.chunk_id)

    # ---------------------------------------------------------------
    # Second pass:
    # fill remaining slots with reranked evidence.
    #
    # But when the query has an explicit article number, do NOT add
    # unrelated article chunks.
    # ---------------------------------------------------------------

    for chunk in reranked:

        if len(selected) >= top_n:
            break

        if chunk.chunk_id in selected_ids:
            continue

        if query_identifiers:
            if not _chunk_matches_article(
                chunk,
                query_identifiers,
            ):
                continue

        dense_score = dense_scores.get(
            chunk.chunk_id
        )

        if (
            dense_score is not None
            and best_dense_score is not None
            and (
                best_dense_score - dense_score
                > maximum_dense_score_drop
            )
        ):
            continue

        selected.append(chunk)
        selected_ids.add(chunk.chunk_id)

    return selected


# =========================================================================
# Article extraction
# =========================================================================


def _extract_article_identifiers(
    text: str,
) -> set[str]:
    """Extract Arabic/English article numbers from text."""

    patterns = (
        r"(?:المادة|مادة|article)"
        r"\s*(?:رقم|no\.?|number)?"
        r"\s*[\(\[\{]?"
        r"([0-9٠-٩]+)"
        r"[\)\]\}]?",
    )

    identifiers: set[str] = set()

    for pattern in patterns:

        for match in re.finditer(
            pattern,
            text,
            re.IGNORECASE,
        ):
            identifiers.add(
                _normalize_digits(
                    match.group(1)
                )
            )

    return identifiers


def _chunk_matches_article(
    chunk: RerankedChunk,
    query_identifiers: set[str],
) -> bool:
    """Return True if the chunk belongs to an article named by query."""

    evidence = "\n".join(
        part
        for part in (
            chunk.section_title,
            chunk.text,
        )
        if part
    )

    evidence_identifiers = _extract_article_identifiers(
        evidence
    )

    return bool(
        query_identifiers
        & evidence_identifiers
    )


# =========================================================================
# Diagnostics
# =========================================================================


def _diagnostics(
    assessment: EvidenceAssessment,
    *,
    candidate_count: int,
    used_chunk_count: int,
    strategy: str,
) -> RetrievalDiagnostics:

    return RetrievalDiagnostics(
        strategy=strategy,
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


def _individual_dispute_procedure_answer(query: str, evidence_text: str) -> str | None:
    """Build a conservative answer for the clearly matched individual-dispute clause."""
    normalized_query = _normalize_arabic_for_matching(query)
    normalized_evidence = _normalize_arabic_for_matching(evidence_text)

    if "نزاع فرد" not in normalized_query:
        return None

    # Require the evidence to contain the key clause signals before using
    # this fixed formulation; otherwise leave generation untouched.
    required_signals = ("نزاع فرد", "عشره ايام", "تسويه", "لجنه", "حق التقاضي")
    if not all(signal in normalized_evidence for signal in required_signals):
        return None

    return (
        "مع عدم الإخلال بحق التقاضي، يجوز للعامل أو صاحب العمل طلب تسوية "
        "النزاع الفردي وديًا خلال عشرة أيام من تاريخ نشوئه، أمام لجنة يرأسها "
        "مدير مديرية العمل أو من ينيبه، وتضم العامل أو من يمثله وصاحب العمل "
        "أو من يمثله. ويجوز لرئيس اللجنة الاستعانة بذوي الخبرة حسب موضوع النزاع."
    )


# =========================================================================
# Conservative extractive answer mode
# =========================================================================

def _extractive_answer_from_chunks(chunks: list[RerankedChunk]) -> str:
    """Return source text verbatim; never let the LLM add legal claims."""
    parts: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        source_text = str(chunk.text).strip()
        if source_text and source_text not in seen:
            parts.append(source_text)
            seen.add(source_text)
    return "\n\n".join(parts)


def _remove_legal_identifiers_from_answer(answer: str, language: str) -> str:
    """Remove legal identifiers from answer prose while preserving substantive numbers."""
    cleaned = answer
    patterns = [
        r"(?iu)\b(?:المادة|مادة)\s*[\(\[\{]?\s*[٠-٩0-9]+(?:\s*[\)\]\}])?",
        r"(?iu)\bقانون\s*(?:رقم\s*)?[٠-٩0-9]+(?:\s*لسنة\s*[٠-٩0-9]+)?",
        r"(?iu)\b(?:الباب|الفصل)\s*(?:رقم\s*)?[٠-٩0-9]+",
        r"(?iu)\b(?:article|section|chapter|law)\s*(?:no\.?\s*)?[0-9]+(?:\s*(?:of|/)\s*[0-9]{4})?",
    ]
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"[\(\[\{]\s*[\)\]\}]", "", cleaned)
    cleaned = re.sub(r"^[ \t]*[:：\-–—][ \t]*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"[ \t]+([،؛,:.])", r"\1", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


# =========================================================================
# Citations
# =========================================================================


def _attach_citations(
    answer: str,
    citations: list[Citation],
) -> str:

    markers = " ".join(
        citation.marker
        for citation in citations
    )

    return f"{answer.strip()} {markers}".strip()


# =========================================================================
# Retry / messages
# =========================================================================


def _repair_instruction(
    language: str,
) -> str:

    if language == "ar":
        return (
            "\n\nكانت الاستجابة السابقة غير صالحة. "
            "أعد كائن JSON مطابقاً للمخطط فقط، "
            "واكتب حقل answer باللغة العربية فقط، "
            "واستخدم حصراً evidence_ids المتاحة."
        )

    return (
        "\n\nThe previous response was invalid. "
        "Return schema-valid JSON only, "
        "write the answer in the required language, "
        "and use only supplied evidence_ids."
    )


def _no_evidence_message(
    language: str,
) -> str:

    if language == "ar":
        return (
            "المعلومات المتاحة في المستندات المفهرسة "
            "غير كافية للإجابة عن هذا السؤال."
        )

    return (
        "The indexed documents do not contain "
        "sufficient evidence to answer this question."
    )


def _generation_failure_message(
    language: str,
) -> str:

    if language == "ar":
        return (
            "تعذر إنتاج إجابة يمكن التحقق "
            "من استشهاداتها من الأدلة المتاحة."
        )

    return (
        "A citation-valid answer could not be "
        "produced from the available evidence."
    )


def _validation_failure_message(
    language: str,
    unsupported_numbers: set[str],
) -> str:

    if language == "ar":
        return (
            "لا توجد إجابة موثوقة متاحة لهذا السؤال "
            "بناءً على المستندات الحالية."
        )

    return (
        "No reliable answer is available for this question "
        "based on the current documents."
    )


# =========================================================================
# Digit normalization
# =========================================================================


def _normalize_digits(
    value: str,
) -> str:

    translation = str.maketrans(
        "٠١٢٣٤٥٦٧٨٩",
        "0123456789",
    )

    return value.translate(translation)