"""Fail-closed evidence sufficiency checks for the RAG pipeline.

The score defaults in this module are experimental smoke-test values, not
production-calibrated legal relevance thresholds. Deployments must calibrate
them against supported queries, paraphrases, identifier lookups, difficult
negatives, and no-support questions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from legal_rag.query.models import RerankedChunk, RetrievedChunk

_IDENTIFIER_PATTERN = re.compile(
    r"(?:المادة|قانون|قرار|مرسوم|article|law|decree|decision)"
    r"\s*(?:رقم|no\.?|number)?\s*[\[(]?[\s:.-]*([0-9٠-٩]+)",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class EvidenceSufficiencyConfig:
    """Configurable, evaluation-calibrated evidence gates."""

    enabled: bool = True
    minimum_dense_score: float = 0.82
    identifier_override_score: float = 0.75
    minimum_rerank_score: float | None = None


@dataclass(frozen=True, slots=True)
class EvidenceAssessment:
    """Observable signals and the resulting sufficiency decision."""

    sufficient: bool
    reason: str
    top_dense_score: float | None
    dense_score_margin: float | None
    top_rerank_score: float | None
    exact_identifier_match: bool
    source_count: int


class EvidenceSufficiencyEvaluator:
    """Evaluate retrieval evidence before any text reaches the LLM."""

    def __init__(self, config: EvidenceSufficiencyConfig | None = None) -> None:
        self._config = config or EvidenceSufficiencyConfig()

    def assess(
        self,
        query: str,
        retrieved: list[RetrievedChunk],
        reranked: list[RerankedChunk],
    ) -> EvidenceAssessment:
        top_dense_score = retrieved[0].score if retrieved else None
        dense_score_margin = retrieved[0].score - retrieved[1].score if len(retrieved) > 1 else None
        top_rerank_score = reranked[0].rerank_score if reranked else None
        exact_identifier_match = _has_exact_identifier_match(query, retrieved)
        source_count = len(
            {chunk.document_id or chunk.source_file or chunk.chunk_id for chunk in retrieved}
        )

        def result(sufficient: bool, reason: str) -> EvidenceAssessment:
            return EvidenceAssessment(
                sufficient=sufficient,
                reason=reason,
                top_dense_score=top_dense_score,
                dense_score_margin=dense_score_margin,
                top_rerank_score=top_rerank_score,
                exact_identifier_match=exact_identifier_match,
                source_count=source_count,
            )

        if not retrieved or not reranked:
            return result(False, "no_retrieved_evidence")

        if not self._config.enabled:
            return result(True, "gate_disabled")

        assert top_dense_score is not None
        dense_sufficient = top_dense_score >= self._config.minimum_dense_score
        identifier_override = (
            exact_identifier_match and top_dense_score >= self._config.identifier_override_score
        )

        if not dense_sufficient and not identifier_override:
            return result(False, "dense_score_below_experimental_threshold")

        minimum_rerank_score = self._config.minimum_rerank_score
        if (
            minimum_rerank_score is not None
            and top_rerank_score is not None
            and top_rerank_score < minimum_rerank_score
        ):
            return result(False, "rerank_score_below_experimental_threshold")

        reason = (
            "exact_identifier_override"
            if identifier_override and not dense_sufficient
            else "sufficient"
        )
        return result(True, reason)


def _has_exact_identifier_match(
    query: str,
    retrieved: list[RetrievedChunk],
) -> bool:
    query_identifiers = {_normalize_digits(value) for value in _IDENTIFIER_PATTERN.findall(query)}
    if not query_identifiers:
        return False

    evidence_text = "\n".join(f"{chunk.section_title or ''}\n{chunk.text}" for chunk in retrieved)
    evidence_identifiers = {
        _normalize_digits(value) for value in _IDENTIFIER_PATTERN.findall(evidence_text)
    }
    return bool(query_identifiers & evidence_identifiers)


def _normalize_digits(value: str) -> str:
    return value.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
