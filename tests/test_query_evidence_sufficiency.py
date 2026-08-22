from legal_rag.query.evidence_sufficiency import (
    EvidenceSufficiencyConfig,
    EvidenceSufficiencyEvaluator,
)
from legal_rag.query.models import RerankedChunk, RetrievedChunk


def _retrieved(chunk_id: str, score: float, text: str = "نص قانوني") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="document-1",
        score=score,
        text=text,
        source_file="law.txt",
    )


def _reranked(chunk: RetrievedChunk, score: float = 1.0) -> RerankedChunk:
    return RerankedChunk(**chunk.__dict__, rerank_score=score)


def test_sufficiency_accepts_supported_smoke_score() -> None:
    retrieved = [_retrieved("article-3", 0.8664), _retrieved("article-2", 0.8366)]
    assessment = EvidenceSufficiencyEvaluator().assess(
        "سؤال مدعوم",
        retrieved,
        [_reranked(retrieved[0])],
    )

    assert assessment.sufficient is True
    assert assessment.reason == "sufficient"
    assert assessment.dense_score_margin is not None


def test_sufficiency_rejects_unsupported_smoke_score() -> None:
    retrieved = [_retrieved("article-3", 0.7828), _retrieved("article-1", 0.7797)]
    assessment = EvidenceSufficiencyEvaluator().assess(
        "سؤال عن إجازة سنوية غير موجودة في المستند",
        retrieved,
        [_reranked(retrieved[0])],
    )

    assert assessment.sufficient is False
    assert assessment.reason == "dense_score_below_experimental_threshold"


def test_exact_legal_identifier_can_use_lower_configured_override() -> None:
    retrieved = [_retrieved("article-3", 0.79, "المادة (3) تحمي بيانات المتعامل")]
    evaluator = EvidenceSufficiencyEvaluator(
        EvidenceSufficiencyConfig(
            minimum_dense_score=0.82,
            identifier_override_score=0.75,
        )
    )

    assessment = evaluator.assess(
        "ما حكم المادة رقم ٣؟",
        retrieved,
        [_reranked(retrieved[0])],
    )

    assert assessment.sufficient is True
    assert assessment.exact_identifier_match is True
    assert assessment.reason == "exact_identifier_override"
