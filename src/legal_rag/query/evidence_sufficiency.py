"""Fail-closed evidence sufficiency checks for the RAG pipeline.

The evaluator decides whether retrieved evidence is strong enough to be
passed to the generation stage.

For legal RAG, dense similarity and raw cross-encoder scores are not
sufficient on their own. The evaluator therefore gives special importance
to explicit legal identifiers such as article numbers and law references.

The system should prefer returning "insufficient evidence" over allowing
the generator to answer from semantically similar but legally unrelated
documents.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from legal_rag.query.models import RerankedChunk, RetrievedChunk


_IDENTIFIER_PATTERN = re.compile(
    r"(?:المادة|مادة|article)"
    r"\s*(?:رقم|no\.?|number)?"
    r"\s*[\(\[\{]?\s*([0-9٠-٩]+)\s*[\)\]\}]?",
    re.IGNORECASE,
)

_LAW_PATTERN = re.compile(
    r"(?:قانون|law|act)\s+([^\n,؛؟?.]+)",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class EvidenceSufficiencyConfig:
    """Configurable evidence gates."""

    enabled: bool = True

    # Initial semantic retrieval gate.
    minimum_dense_score: float = 0.82

    # Allows an explicit identifier match to rescue a slightly weaker
    # dense result.
    identifier_override_score: float = 0.75

    # Optional raw cross-encoder threshold.
    #
    # IMPORTANT:
    # This is intentionally optional. A negative raw score does NOT
    # automatically mean that legal evidence is insufficient.
    minimum_rerank_score: float | None = None

    # Minimum lexical overlap for generic questions where no explicit
    # article identifier exists.
    minimum_lexical_overlap: float = 0.08


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

    def __init__(
        self,
        config: EvidenceSufficiencyConfig | None = None,
    ) -> None:
        self._config = config or EvidenceSufficiencyConfig()

    def assess(
        self,
        query: str,
        retrieved: list[RetrievedChunk],
        reranked: list[RerankedChunk],
    ) -> EvidenceAssessment:

        top_dense_score = (
            retrieved[0].score
            if retrieved
            else None
        )

        dense_score_margin = (
            retrieved[0].score - retrieved[1].score
            if len(retrieved) > 1
            else None
        )

        top_rerank_score = (
            reranked[0].rerank_score
            if reranked
            else None
        )

        source_count = len(
            {
                chunk.document_id
                or chunk.source_file
                or chunk.chunk_id
                for chunk in retrieved
            }
        )

        # ---------------------------------------------------------------
        # Query identifiers
        # ---------------------------------------------------------------

        query_identifiers = _extract_identifiers(query)

        # IMPORTANT:
        # Check the reranked evidence first, not all retrieved candidates.
        #
        # The fact that one of the 20 candidates contains المادة (8)
        # does not mean the selected evidence supports the answer.
        # ---------------------------------------------------------------

        exact_identifier_match = _has_exact_identifier_match_in_reranked(
            query,
            reranked,
        )

        law_match = _has_law_match(
            query,
            reranked,
        )

        lexical_overlap = _best_lexical_overlap(
            query,
            reranked,
        )

        def result(
            sufficient: bool,
            reason: str,
        ) -> EvidenceAssessment:
            return EvidenceAssessment(
                sufficient=sufficient,
                reason=reason,
                top_dense_score=top_dense_score,
                dense_score_margin=dense_score_margin,
                top_rerank_score=top_rerank_score,
                exact_identifier_match=exact_identifier_match,
                source_count=source_count,
            )

        # ---------------------------------------------------------------
        # Basic retrieval failure
        # ---------------------------------------------------------------

        if not retrieved or not reranked:
            return result(
                False,
                "no_retrieved_evidence",
            )

        if not self._config.enabled:
            return result(
                True,
                "gate_disabled",
            )

        assert top_dense_score is not None

        # ---------------------------------------------------------------
        # Dense retrieval gate
        # ---------------------------------------------------------------

        dense_sufficient = (
            top_dense_score >= self._config.minimum_dense_score
        )

        identifier_override = (
            bool(query_identifiers)
            and exact_identifier_match
            and top_dense_score >= self._config.identifier_override_score
        )

        if not dense_sufficient and not identifier_override:
            return result(
                False,
                "dense_score_below_experimental_threshold",
            )

        # ---------------------------------------------------------------
        # CASE 1:
        #
        # Explicit article identifier exists in the question and the
        # selected evidence contains the same article.
        #
        # This is the strongest legal retrieval signal.
        # ---------------------------------------------------------------

        if query_identifiers and exact_identifier_match:

            # If the query explicitly names a law and we have a law match,
            # this is even stronger.
            if law_match:
                return result(
                    True,
                    "exact_article_and_law_match",
                )

            return result(
                True,
                "exact_article_match",
            )

        # ---------------------------------------------------------------
        # CASE 2:
        #
        # Generic question without an explicit article number.
        #
        # We need meaningful lexical overlap before allowing generation.
        # ---------------------------------------------------------------

        if (
    lexical_overlap < self._config.minimum_lexical_overlap
    and not (
        top_rerank_score is not None
        and top_rerank_score >= 5.0
           )
          ):
         return result(
        False,
           "insufficient_lexical_support",
         )

        # ---------------------------------------------------------------
        # Optional raw rerank threshold.
        #
        # We only use this for cases where there is no strong legal
        # identifier match.
        #
        # A negative raw cross-encoder score is NOT inherently a failure.
        # ---------------------------------------------------------------

        minimum_rerank_score = self._config.minimum_rerank_score

        if (
            minimum_rerank_score is not None
            and top_rerank_score is not None
            and top_rerank_score < minimum_rerank_score
        ):
            return result(
                False,
                "rerank_score_below_experimental_threshold",
            )

        return result(
            True,
            "sufficient",
        )


# ---------------------------------------------------------------------------
# Identifier matching
# ---------------------------------------------------------------------------


def _extract_identifiers(text: str) -> set[str]:
    """Extract explicit article identifiers."""

    return {
        _normalize_digits(match.group(1))
        for match in _IDENTIFIER_PATTERN.finditer(text)
    }


def _has_exact_identifier_match_in_reranked(
    query: str,
    reranked: list[RerankedChunk],
) -> bool:
    """Return True only when the selected evidence contains the same
    article identifier as the query.
    """

    query_identifiers = _extract_identifiers(query)

    if not query_identifiers:
        return False

    # Only inspect the evidence that survived reranking.
    for chunk in reranked:
        evidence_text = "\n".join(
            part
            for part in (
                chunk.section_title,
                chunk.text,
            )
            if part
        )

        evidence_identifiers = _extract_identifiers(
            evidence_text
        )

        if query_identifiers & evidence_identifiers:
            return True

    return False


# ---------------------------------------------------------------------------
# Law matching
# ---------------------------------------------------------------------------


def _extract_law_terms(text: str) -> set[str]:
    """Extract meaningful words following an explicit law marker."""

    normalized = _normalize_text(text)

    terms: set[str] = set()

    for match in _LAW_PATTERN.finditer(normalized):
        phrase = match.group(1)

        # Stop at common question/legal separators.
        phrase = re.split(
            r"\b(?:المادة|مادة|حكم|العقوبة|العقوبات|الجزاء|ما|ماذا)\b",
            phrase,
            maxsplit=1,
        )[0]

        for token in phrase.split():
            token = token.strip()

            if len(token) >= 3:
                terms.add(token)

    return terms


def _has_law_match(
    query: str,
    reranked: list[RerankedChunk],
) -> bool:
    """Check whether selected evidence supports the law named in query."""

    query_terms = _extract_law_terms(query)

    if not query_terms:
        # Query does not explicitly name a law.
        return False

    for chunk in reranked:
        evidence = "\n".join(
            part
            for part in (
                chunk.document_id,
                chunk.source_file,
                chunk.section_title,
                chunk.text,
            )
            if part
        )

        evidence_terms = _extract_law_terms(evidence)

        if not evidence_terms:
            continue

        overlap = query_terms & evidence_terms

        # At least one meaningful law-name term is enough here because
        # OCR can damage Arabic legal document names.
        if overlap:
            return True

    return False


# ---------------------------------------------------------------------------
# Lexical support
# ---------------------------------------------------------------------------


def _best_lexical_overlap(
    query: str,
    reranked: list[RerankedChunk],
) -> float:
    """Return the strongest lightweight lexical overlap."""

    if not reranked:
        return 0.0

    query_tokens = _meaningful_tokens(query)

    if not query_tokens:
        return 0.0

    best = 0.0

    for chunk in reranked:
        evidence = "\n".join(
            part
            for part in (
                chunk.section_title,
                chunk.text,
            )
            if part
        )

        evidence_tokens = _meaningful_tokens(evidence)

        overlap = len(
            query_tokens & evidence_tokens
        )

        score = overlap / max(
            len(query_tokens),
            1,
        )

        best = max(best, score)

    return best


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------


def _meaningful_tokens(text: str) -> set[str]:
    normalized = _normalize_text(text)

    stop_words = {
        "ما",
        "هي",
        "هو",
        "من",
        "في",
        "على",
        "عن",
        "الى",
        "إلى",
        "هذا",
        "هذه",
        "ذلك",
        "تلك",
        "هل",
        "كيف",
        "ماذا",
        "و",
        "او",
        "أو",
        "the",
        "what",
        "is",
        "are",
        "of",
        "in",
        "on",
        "for",
        "to",
        "and",
    }

    return {
        token
        for token in normalized.split()
        if len(token) >= 3
        and token not in stop_words
    }


def _normalize_text(text: str) -> str:
    text = _normalize_digits(text)

    # Remove Arabic diacritics.
    text = re.sub(
        r"[\u064B-\u065F\u0670]",
        "",
        text,
    )

    # Normalize common Arabic OCR variants.
    text = text.replace("أ", "ا")
    text = text.replace("إ", "ا")
    text = text.replace("آ", "ا")
    text = text.replace("ى", "ي")
    text = text.replace("ة", "ه")

    # Normalize punctuation.
    text = re.sub(
        r"[\(\)\[\]\{\}:،,؛;.!؟?\"'«»ـ\-_/\\]+",
        " ",
        text,
    )

    # Collapse whitespace.
    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip().lower()


def _normalize_digits(value: str) -> str:
    return value.translate(
        str.maketrans(
            "٠١٢٣٤٥٦٧٨٩",
            "0123456789",
        )
    )