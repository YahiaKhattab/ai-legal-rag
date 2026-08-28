"""Legal-aware cross-encoder reranking.

The initial Qdrant vector search retrieves a broad candidate set.
The cross-encoder then scores query/chunk pairs jointly.

For legal RAG, semantic similarity alone is not enough:
a chunk from a different law can look highly similar to the query.
This module therefore combines the cross-encoder score with explicit
legal signals such as:

- exact article/section identifier matches
- explicit law-name matches
- important query-term overlap
- penalties for conflicting law identifiers

The original cross-encoder score is preserved in ``rerank_score``.
The additional legal-aware score is used only for ordering.
"""

from __future__ import annotations

import re
from functools import lru_cache

from sentence_transformers import CrossEncoder

from legal_rag.query.models import RerankedChunk, RetrievedChunk


_DEFAULT_MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

# Common Arabic prefixes that should not prevent useful matching.
_ARABIC_PREFIXES = (
    "وال",
    "بال",
    "كال",
    "لل",
    "ال",
)


def _normalize_digits(text: str) -> str:
    return text.translate(_ARABIC_DIGITS)


def _normalize_text(text: str) -> str:
    """Normalize Arabic/English text for matching only.

    This does NOT modify the actual evidence sent to the LLM.
    """

    text = _normalize_digits(text)

    # Arabic presentation forms / punctuation are common in OCR output.
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)

    # Normalize common Arabic letter variants.
    text = text.replace("أ", "ا")
    text = text.replace("إ", "ا")
    text = text.replace("آ", "ا")
    text = text.replace("ى", "ي")
    text = text.replace("ة", "ه")

    # Normalize punctuation.
    text = re.sub(r"[\(\)\[\]\{\}:،,؛;.!؟?\"'«»ـ\-_/\\]+", " ", text)

    # Collapse whitespace.
    text = re.sub(r"\s+", " ", text).strip().lower()

    return text


def _tokens(text: str) -> set[str]:
    normalized = _normalize_text(text)

    tokens: set[str] = set()

    for token in normalized.split():
        token = token.strip()

        if not token:
            continue

        # Keep numbers intact.
        if token.isdigit():
            tokens.add(token)
            continue

        # Add original token.
        tokens.add(token)

        # Also add a light prefix-stripped representation.
        for prefix in _ARABIC_PREFIXES:
            if token.startswith(prefix) and len(token) > len(prefix) + 2:
                tokens.add(token[len(prefix):])
                break

    return tokens


# ---------------------------------------------------------------------------
# Legal identifier extraction
# ---------------------------------------------------------------------------

_ARTICLE_PATTERNS = (
    re.compile(
        r"(?:المادة|مادة|article)"
        r"\s*(?:رقم|رقم\s*المادة|no\.?|number)?"
        r"\s*[\(\[\{]?\s*([0-9]+)\s*[\)\]\}]?",
        re.IGNORECASE,
    ),
)


def _extract_article_numbers(text: str) -> set[str]:
    normalized = _normalize_digits(text)
    numbers: set[str] = set()

    for pattern in _ARTICLE_PATTERNS:
        for match in pattern.finditer(normalized):
            numbers.add(match.group(1))

    return numbers


def _extract_law_phrase(query: str) -> str | None:
    """Extract a likely law/document name from the query.

    This intentionally stays conservative. We only use phrases that
    explicitly contain legal-document markers such as 'قانون'.
    """

    normalized = _normalize_text(query)

    # Arabic:
    match = re.search(
        r"(قانون\s+[^\n,؛؟?.]+?)(?:\s+(?:ما|ماذا|هي|هو|المادة|حكم|العقوبة|العقوبات|الجزاء)\b|$)",
        normalized,
    )
    if match:
        phrase = match.group(1).strip()
        phrase = re.sub(r"\s+", " ", phrase)
        if len(phrase) >= 8:
            return phrase

    # English:
    match = re.search(
        r"\b((?:law|act)\s+[^,.;?]+)",
        normalized,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()

    return None


def _law_name_matches(query: str, candidate: RetrievedChunk) -> bool:
    law_phrase = _extract_law_phrase(query)

    if not law_phrase:
        return False

    query_tokens = _tokens(law_phrase)

    candidate_metadata = " ".join(
        part
        for part in (
            candidate.document_id,
            candidate.source_file,
            candidate.section_title,
            candidate.text,
        )
        if part
    )

    candidate_tokens = _tokens(candidate_metadata)

    # Require at least two meaningful law-name tokens to match.
    meaningful = {
        token
        for token in query_tokens
        if len(token) >= 3 and not token.isdigit()
    }

    if not meaningful:
        return False

    return len(meaningful & candidate_tokens) >= min(2, len(meaningful))


def _has_conflicting_law(query: str, candidate: RetrievedChunk) -> bool:
    """Detect obvious cross-document law conflicts.

    Example:
        Query -> قانون حماية المستهلك
        Candidate -> قانون رعاية حقوق المسنين

    We do not reject the candidate here because metadata can be imperfect;
    instead we apply a strong ranking penalty.
    """

    law_phrase = _extract_law_phrase(query)

    if not law_phrase:
        return False

    query_tokens = {
        token
        for token in _tokens(law_phrase)
        if len(token) >= 3 and token not in {"قانون"}
    }

    if not query_tokens:
        return False

    candidate_metadata = " ".join(
        part
        for part in (
            candidate.document_id,
            candidate.source_file,
            candidate.section_title,
        )
        if part
    )

    candidate_text = _normalize_text(candidate_metadata)

    # Explicitly recognized legal-document phrase.
    candidate_law_match = re.search(
        r"(?:قانون|law|act)\s+[^\n,؛؟?.]+",
        candidate_text,
        re.IGNORECASE,
    )

    if not candidate_law_match:
        return False

    candidate_law_tokens = {
        token
        for token in _tokens(candidate_law_match.group(0))
        if len(token) >= 3 and token not in {"قانون", "law", "act"}
    }

    if not candidate_law_tokens:
        return False

    # A clearly different law name is a conflict.
    overlap = query_tokens & candidate_law_tokens

    return not overlap


# ---------------------------------------------------------------------------
# Query/evidence matching
# ---------------------------------------------------------------------------

def _article_match_score(
    query: str,
    candidate: RetrievedChunk,
) -> float:
    query_articles = _extract_article_numbers(query)

    if not query_articles:
        return 0.0

    evidence_text = "\n".join(
        part
        for part in (
            candidate.section_title,
            candidate.text,
        )
        if part
    )

    evidence_articles = _extract_article_numbers(evidence_text)

    if not evidence_articles:
        return 0.0

    if query_articles & evidence_articles:
        return 1.0

    return -1.0


def _important_term_overlap(
    query: str,
    candidate: RetrievedChunk,
) -> float:
    """Measure useful lexical overlap.

    This is deliberately lightweight. It is not intended to replace the
    embedding model or cross-encoder.
    """

    query_tokens = {
        token
        for token in _tokens(query)
        if len(token) >= 3 and not token.isdigit()
    }

    if not query_tokens:
        return 0.0

    evidence_text = " ".join(
        part
        for part in (
            candidate.section_title,
            candidate.text,
        )
        if part
    )

    evidence_tokens = _tokens(evidence_text)

    overlap = len(query_tokens & evidence_tokens)

    return min(overlap / max(len(query_tokens), 1), 1.0)


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------

class CrossEncoderReranker:
    """Cross-encoder reranker with legal-aware scoring."""

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL_NAME,
        device: str | None = None,
    ) -> None:
        self._model = CrossEncoder(
            model_name,
            device=device,
            max_length=512,
        )

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_n: int = 6,
    ) -> list[RerankedChunk]:
        if not candidates:
            return []

        # ---------------------------------------------------------------
        # 1. Cross-encoder scores
        # ---------------------------------------------------------------

        pairs = [
            (query, candidate.text)
            for candidate in candidates
        ]

        scores = self._model.predict(pairs)

        reranked: list[tuple[RerankedChunk, float]] = []

        # ---------------------------------------------------------------
        # 2. Combine semantic + legal signals
        # ---------------------------------------------------------------

        for candidate, raw_score in zip(
            candidates,
            scores,
            strict=True,
        ):
            raw_score = float(raw_score)

            article_score = _article_match_score(
                query,
                candidate,
            )

            law_match = _law_name_matches(
                query,
                candidate,
            )

            conflicting_law = _has_conflicting_law(
                query,
                candidate,
            )

            lexical_overlap = _important_term_overlap(
                query,
                candidate,
            )

            # Start from the original cross-encoder score.
            final_score = raw_score

            # -----------------------------------------------------------
            # Strong article match.
            #
            # A question asking about المادة (8) should strongly prefer
            # evidence explicitly mentioning المادة (8).
            # -----------------------------------------------------------

            if article_score > 0:
                final_score += 2.50

            elif article_score < 0:
                # Only a mild penalty: a chunk may legitimately contain
                # the answer without explicitly repeating the article.
                final_score -= 0.35

            # -----------------------------------------------------------
            # Explicit law-name match.
            # -----------------------------------------------------------

            if law_match:
                final_score += 1.50

            # -----------------------------------------------------------
            # Conflicting law/document.
            #
            # This is intentionally stronger than lexical similarity.
            # -----------------------------------------------------------

            if conflicting_law:
                final_score -= 2.00

            # -----------------------------------------------------------
            # Useful lexical overlap.
            # -----------------------------------------------------------

            final_score += 0.75 * lexical_overlap

            # Store the ORIGINAL cross-encoder score in rerank_score.
            #
            # Other parts of the existing pipeline may use this field
            # for diagnostics / thresholds, so we do not silently change
            # its meaning.
            reranked_chunk = RerankedChunk(
                **candidate.__dict__,
                rerank_score=raw_score,
            )

            reranked.append(
                (
                    reranked_chunk,
                    final_score,
                )
            )

        # ---------------------------------------------------------------
        # 3. Sort using legal-aware score
        # ---------------------------------------------------------------

        reranked.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        return [
            chunk
            for chunk, _final_score in reranked[:top_n]
        ]


@lru_cache(maxsize=1)
def get_default_reranker() -> CrossEncoderReranker:
    return CrossEncoderReranker()