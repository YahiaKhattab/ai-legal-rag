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
    retrieved = [
        _retrieved("article-3", 0.8664, "لا يجوز استخدام البيانات دون موافقة"),
        _retrieved("article-2", 0.8366),
    ]
    assessment = EvidenceSufficiencyEvaluator().assess(
        "هل يجوز استخدام البيانات دون موافقة؟",
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
    assert assessment.reason == "exact_article_match"


def test_positive_reranker_can_rescue_supported_dense_near_miss() -> None:
    retrieved = [
        _retrieved(
            "article-1",
            0.8144,
            "يجب التحقق من هوية المتعامل قبل تفعيل الخدمة",
        )
    ]
    evaluator = EvidenceSufficiencyEvaluator(EvidenceSufficiencyConfig(minimum_rerank_score=-1.0))

    assessment = evaluator.assess(
        "ما الإجراء الواجب للتحقق من هوية المتعامل قبل تفعيل الخدمة؟",
        retrieved,
        [_reranked(retrieved[0], score=2.33)],
    )

    assert assessment.sufficient is True
    assert assessment.reason == "rerank_score_override"


def test_negative_reranker_rejects_close_topic_false_positive() -> None:
    retrieved = [
        _retrieved(
            "article-3",
            0.8268,
            "يجب عرض الرسوم قبل إبرام التعاقد الإلكتروني",
        )
    ]
    evaluator = EvidenceSufficiencyEvaluator(EvidenceSufficiencyConfig(minimum_rerank_score=-1.0))

    assessment = evaluator.assess(
        "هل يحق إلغاء التعاقد خلال أربعة عشر يوما واسترداد الرسوم؟",
        retrieved,
        [_reranked(retrieved[0], score=-2.46)],
    )

    assert assessment.sufficient is False
    assert assessment.reason == "rerank_score_below_experimental_threshold"


def test_missing_explicit_article_is_rejected() -> None:
    retrieved = [_retrieved("article-3", 0.8311, "المادة (3): أحكام التعاقد")]
    evaluator = EvidenceSufficiencyEvaluator(EvidenceSufficiencyConfig(minimum_rerank_score=-1.0))

    assessment = evaluator.assess(
        "ما حقوق المتعامل وفقا للمادة رقم ٩٩؟",
        retrieved,
        [_reranked(retrieved[0], score=0.71)],
    )

    assert assessment.sufficient is False
    assert assessment.reason == "explicit_article_identifier_not_found"
