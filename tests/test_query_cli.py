from dataclasses import replace
from typing import cast

import pytest

import legal_rag.query.cli as cli_module
from legal_rag.config import Settings
from legal_rag.query.models import CitedAnswer, RetrievalDiagnostics
from legal_rag.query.pipeline import RAGAnswerPipeline
from legal_rag.query.retriever import RetrievalFilters


class FakePipeline:
    def answer(
        self,
        query: str,
        language: str = "mixed",
        filters: RetrievalFilters | None = None,
    ) -> CitedAnswer:
        assert query == "سؤال"
        assert language == "ar"
        assert filters is not None
        assert filters.source == "official"
        return CitedAnswer(
            query=query,
            answer_text="المعلومات غير كافية.",
            language=language,
            citations=[],
            retrieved_chunk_ids=["chunk-1"],
            retrieval=RetrievalDiagnostics(
                strategy="dense_plus_cross_encoder",
                candidate_count=1,
                used_chunk_count=0,
                sufficient=False,
                reason="dense_score_below_experimental_threshold",
                top_dense_score=0.78,
                dense_score_margin=None,
                top_rerank_score=-1.0,
                exact_identifier_match=False,
                source_count=1,
            ),
        )


class GenerationFailurePipeline(FakePipeline):
    def answer(
        self,
        query: str,
        language: str = "mixed",
        filters: RetrievalFilters | None = None,
    ) -> CitedAnswer:
        result = super().answer(query, language, filters)
        assert result.retrieval is not None
        return replace(
            result,
            answer_text="تعذر إنتاج إجابة قابلة للتحقق.",
            retrieval=replace(
                result.retrieval,
                reason="invalid_structured_generation",
            ),
        )


def test_query_cli_prints_fail_closed_retrieval_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_pipeline = FakePipeline()

    def fake_build_pipeline(
        settings: Settings,
        *,
        retrieve_top_k: int,
        rerank_top_n: int,
    ) -> RAGAnswerPipeline:
        del settings
        assert retrieve_top_k == 10
        assert rerank_top_n == 4
        return cast(RAGAnswerPipeline, fake_pipeline)

    monkeypatch.setattr(cli_module, "_build_pipeline", fake_build_pipeline)
    monkeypatch.setattr(
        "sys.argv",
        [
            "legal-rag-query",
            "سؤال",
            "--language",
            "ar",
            "--top-k",
            "10",
            "--top-n",
            "4",
            "--source",
            "official",
        ],
    )

    cli_module.main()

    output = capsys.readouterr().out
    assert "Decision: insufficient" in output
    assert "Top dense score: 0.7800" in output
    assert "المعلومات غير كافية" in output


def test_query_cli_distinguishes_generation_validation_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_pipeline = GenerationFailurePipeline()

    def fake_build_pipeline(
        settings: Settings,
        *,
        retrieve_top_k: int,
        rerank_top_n: int,
    ) -> RAGAnswerPipeline:
        del settings, retrieve_top_k, rerank_top_n
        return cast(RAGAnswerPipeline, fake_pipeline)

    monkeypatch.setattr(
        cli_module,
        "_build_pipeline",
        fake_build_pipeline,
    )
    monkeypatch.setattr(
        "sys.argv",
        ["legal-rag-query", "سؤال", "--language", "ar", "--source", "official"],
    )

    cli_module.main()

    output = capsys.readouterr().out
    assert "Decision: generation_failure" in output
    assert "answer validation failed" in output
    assert "Insufficient legal evidence" not in output
