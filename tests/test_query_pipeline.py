from collections.abc import Mapping

from legal_rag.query.evidence_sufficiency import EvidenceSufficiencyEvaluator
from legal_rag.query.models import RerankedChunk, RetrievedChunk
from legal_rag.query.pipeline import RAGAnswerPipeline
from legal_rag.query.retriever import RetrievalFilters


def _chunk(chunk_id: str, score: float, text: str, section: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="document-1",
        score=score,
        text=text,
        source_file="legal_fintech_test_ar.txt",
        section_title=section,
        source="baseline-fintech",
    )


class FakeRetriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = chunks

    def search(
        self,
        query: str,
        top_k: int = 20,
        filters: RetrievalFilters | None = None,
        score_threshold: float | None = None,
    ) -> list[RetrievedChunk]:
        del query, filters, score_threshold
        return self._chunks[:top_k]


class ReversingReranker:
    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_n: int = 6,
    ) -> list[RerankedChunk]:
        del query
        return [
            RerankedChunk(**chunk.__dict__, rerank_score=float(index))
            for index, chunk in enumerate(reversed(candidates[:top_n]), start=1)
        ]


class FakeGenerator:
    def __init__(self, responses: list[str]) -> None:
        self._responses = iter(responses)
        self.calls: list[tuple[str, str | None, Mapping[str, object] | None]] = []

    def generate(
        self,
        prompt: str,
        temperature: float = 0.1,
        *,
        system: str | None = None,
        format_schema: Mapping[str, object] | None = None,
    ) -> str:
        del temperature
        self.calls.append((prompt, system, format_schema))
        return next(self._responses)


def test_pipeline_preserves_dense_top_and_uses_validated_marker() -> None:
    article_3 = _chunk(
        "article-3",
        0.8664,
        "المادة (3): يجب عرض الرسوم ولا يجوز استخدام البيانات دون موافقة.",
        "المادة (3)",
    )
    article_1 = _chunk(
        "article-1",
        0.8366,
        "المادة (1): يجب التحقق من الهوية.",
        "المادة (1)",
    )
    generator = FakeGenerator(
        [
            '{"answer":"استخدام البيانات دون موافقة مخالفة.",'
            '"evidence_ids":["E1"],"insufficient_evidence":false}'
        ]
    )
    pipeline = RAGAnswerPipeline(
        retriever=FakeRetriever([article_3, article_1]),
        reranker=ReversingReranker(),
        generator=generator,
        sufficiency_evaluator=EvidenceSufficiencyEvaluator(),
        evidence_top_n=2,
    )

    result = pipeline.answer("هل يجوز استخدام البيانات دون موافقة؟", language="ar")

    assert result.answer_text.endswith("[1]")
    assert [citation.chunk_id for citation in result.citations] == ["article-3"]
    assert [excerpt.chunk_id for excerpt in result.legal_excerpts] == ["article-3"]
    assert generator.calls[0][1] is not None
    assert generator.calls[0][2] is not None
    assert "يجب التحقق من الهوية" not in generator.calls[0][0]
    assert "هل يجوز استخدام البيانات دون موافقة؟" in generator.calls[0][0]


def test_pipeline_rejects_weak_evidence_without_calling_generator() -> None:
    generator = FakeGenerator([])
    weak = _chunk("article-3", 0.7828, "نص عن التكنولوجيا المالية", "المادة (3)")
    pipeline = RAGAnswerPipeline(
        retriever=FakeRetriever([weak]),
        reranker=ReversingReranker(),
        generator=generator,
    )

    result = pipeline.answer("ما حكم الإجازة السنوية؟", language="ar")

    assert result.citations == []
    assert result.legal_excerpts == []
    assert result.retrieval is not None
    assert result.retrieval.sufficient is False
    assert result.retrieval.reason == "dense_score_below_experimental_threshold"
    assert generator.calls == []


def test_unknown_evidence_marker_retries_once_then_fails_closed() -> None:
    strong = _chunk("article-3", 0.9, "المادة (3): نص قانوني", "المادة (3)")
    invalid = '{"answer":"إجابة","evidence_ids":["E99"],"insufficient_evidence":false}'
    generator = FakeGenerator([invalid, invalid])
    pipeline = RAGAnswerPipeline(
        retriever=FakeRetriever([strong]),
        reranker=ReversingReranker(),
        generator=generator,
        generation_retry_count=1,
    )

    result = pipeline.answer("ما حكم المادة؟", language="ar")

    assert len(generator.calls) == 2
    assert result.citations == []
    assert result.retrieval is not None
    assert result.retrieval.reason == "invalid_structured_generation"


def test_model_generated_citation_marker_is_rejected() -> None:
    strong = _chunk("article-3", 0.9, "المادة (3): نص قانوني", "المادة (3)")
    invalid = '{"answer":"إجابة [1]","evidence_ids":["E1"],"insufficient_evidence":false}'
    valid = '{"answer":"إجابة","evidence_ids":["E1"],"insufficient_evidence":false}'
    generator = FakeGenerator([invalid, valid])
    pipeline = RAGAnswerPipeline(
        retriever=FakeRetriever([strong]),
        reranker=ReversingReranker(),
        generator=generator,
        generation_retry_count=1,
    )

    result = pipeline.answer("ما حكم المادة؟", language="ar")

    assert len(generator.calls) == 2
    assert result.answer_text == "إجابة [1]"
    assert len(result.citations) == 1


def test_selected_evidence_is_rendered_with_sequential_application_markers() -> None:
    first = _chunk("article-3", 0.9, "المادة (3): النص الأول", "المادة (3)")
    second = _chunk("article-1", 0.895, "المادة (1): النص الثاني", "المادة (1)")
    generator = FakeGenerator(
        ['{"answer":"إجابة","evidence_ids":["E2"],"insufficient_evidence":false}']
    )
    pipeline = RAGAnswerPipeline(
        retriever=FakeRetriever([first, second]),
        reranker=ReversingReranker(),
        generator=generator,
        evidence_top_n=2,
    )

    result = pipeline.answer("ما حكم النص الثاني؟", language="ar")

    assert result.answer_text == "إجابة [1]"
    assert result.citations[0].marker == "[1]"


def test_wrong_answer_language_retries_then_accepts_arabic() -> None:
    strong = _chunk("article-3", 0.9, "المادة (3): نص قانوني", "المادة (3)")
    chinese = '{"answer":"公司违反了第三条","evidence_ids":["E1"],"insufficient_evidence":false}'
    arabic = (
        '{"answer":"تخالف الشركة أحكام المادة الثالثة","evidence_ids":["E1"],'
        '"insufficient_evidence":false}'
    )
    generator = FakeGenerator([chinese, arabic])
    pipeline = RAGAnswerPipeline(
        retriever=FakeRetriever([strong]),
        reranker=ReversingReranker(),
        generator=generator,
        generation_retry_count=1,
    )

    result = pipeline.answer("ما حكم المادة؟", language="ar")

    assert len(generator.calls) == 2
    assert result.answer_text == "تخالف الشركة أحكام المادة الثالثة [1]"
    assert "باللغة العربية فقط" in generator.calls[1][0]
