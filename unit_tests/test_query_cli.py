"""Unit tests for legal_rag.query.cli.main.

The CLI wires together argument parsing + RAGAnswerPipeline + printing to
stdout. We patch RAGAnswerPipeline itself (imported into the cli module)
so no real retriever/reranker/generator ever gets constructed, and we
patch sys.argv to simulate command-line invocation.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from legal_rag.query import cli
from legal_rag.query.models import Citation, CitedAnswer, LegalExcerpt


def _fake_answer(**overrides):
    defaults = dict(
        query="q",
        answer_text="The answer is X. [1]",
        language="en",
        citations=[
            Citation(
                marker="[1]",
                chunk_id="c1",
                document_id="d1",
                source_file="law.pdf",
                section_title="Article 5",
                page=3,
            )
        ],
        retrieved_chunk_ids=["c1"],
        legal_excerpts=[
            LegalExcerpt(
                marker="[1]",
                text="Legal excerpt text.",
                source_file="law.pdf",
                section_title="Article 5",
                page=3,
                chunk_id="c1",
            )
        ],
    )
    defaults.update(overrides)
    return CitedAnswer(**defaults)


def test_main_prints_answer_and_sources(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["legal-rag-query", "What is article 5?"])

    fake_pipeline_instance = MagicMock()
    fake_pipeline_instance.answer.return_value = _fake_answer()

    with patch.object(cli, "RAGAnswerPipeline", return_value=fake_pipeline_instance), patch.object(
        cli, "LegalRetriever", return_value=MagicMock()
    ):
        cli.main()

    output = capsys.readouterr().out
    assert "Answer:" in output
    assert "The answer is X." in output
    assert "Legal Evidence:" in output
    assert "Sources:" in output
    assert "law.pdf" in output


def test_main_passes_parsed_arguments_to_the_pipeline(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["legal-rag-query", "my question", "--language", "ar", "--top-k", "5", "--top-n", "2"],
    )

    fake_pipeline_instance = MagicMock()
    fake_pipeline_instance.answer.return_value = _fake_answer(language="ar")
    captured_kwargs = {}

    def fake_pipeline_constructor(**kwargs):
        captured_kwargs.update(kwargs)
        return fake_pipeline_instance

    with patch.object(cli, "RAGAnswerPipeline", side_effect=fake_pipeline_constructor), patch.object(
        cli, "LegalRetriever", return_value=MagicMock()
    ):
        cli.main()

    assert captured_kwargs["retrieve_top_k"] == 5
    assert captured_kwargs["rerank_top_n"] == 2
    fake_pipeline_instance.answer.assert_called_once()
    call_kwargs = fake_pipeline_instance.answer.call_args
    assert call_kwargs.kwargs["language"] == "ar"


def test_main_omits_evidence_and_sources_sections_when_empty(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["legal-rag-query", "q"])

    fake_pipeline_instance = MagicMock()
    fake_pipeline_instance.answer.return_value = _fake_answer(citations=[], legal_excerpts=[])

    with patch.object(cli, "RAGAnswerPipeline", return_value=fake_pipeline_instance), patch.object(
        cli, "LegalRetriever", return_value=MagicMock()
    ):
        cli.main()

    output = capsys.readouterr().out
    assert "Legal Evidence:" not in output
    assert "Sources:" not in output
