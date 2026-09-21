"""Unit tests for legal_rag.query.pipeline.RAGAnswerPipeline.

This orchestrates: retriever -> diversify -> reranker -> prompt_builder
-> generator -> answer_validator. A unit test replaces the retriever,
reranker, and generator with fakes/mocks so we test the *orchestration
logic* (what happens in what order, and how failures are handled)
without depending on Qdrant, a cross-encoder model, or a running Ollama
server.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from legal_rag.query.models import RerankedChunk, RetrievedChunk
from legal_rag.query.pipeline import RAGAnswerPipeline, _diversify_candidates


def _retrieved(chunk_id, document_id="doc-1", text="Some legal text.", score=0.9):
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        score=score,
        text=text,
        source_file="law.pdf",
        page=1,
    )


def _reranked(chunk_id, document_id="doc-1", text="Some legal text.", rerank_score=0.5):
    return RerankedChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        score=0.9,
        text=text,
        source_file="law.pdf",
        page=1,
        rerank_score=rerank_score,
    )


def _pipeline(*, retriever=None, reranker=None, generator=None):
    return RAGAnswerPipeline(
        retriever=retriever or MagicMock(),
        reranker=reranker or MagicMock(),
        generator=generator or MagicMock(),
    )


def test_returns_no_evidence_message_when_retriever_finds_nothing():
    retriever = MagicMock()
    retriever.search.return_value = []
    pipeline = _pipeline(retriever=retriever)

    result = pipeline.answer("What is the rule?", language="en")

    assert "No relevant legal excerpts" in result.answer_text
    assert result.citations == []
    assert result.retrieved_chunk_ids == []


def test_returns_arabic_no_evidence_message_when_language_is_arabic():
    retriever = MagicMock()
    retriever.search.return_value = []
    pipeline = _pipeline(retriever=retriever)

    result = pipeline.answer("سؤال", language="ar")

    assert "لم يتم العثور" in result.answer_text


def test_happy_path_generates_answer_and_attaches_citation():
    retrieved = [_retrieved("c1")]
    reranked = [_reranked("c1", text="The appeal period is thirty days.")]

    retriever = MagicMock()
    retriever.search.return_value = retrieved
    reranker = MagicMock()
    reranker.rerank.return_value = reranked
    generator = MagicMock()
    generator.generate.return_value = "The appeal period is thirty days."

    pipeline = _pipeline(retriever=retriever, reranker=reranker, generator=generator)

    result = pipeline.answer("How long is the appeal period?", language="en")

    assert result.answer_text.endswith("[1]")
    assert len(result.citations) == 1
    assert result.citations[0].chunk_id == "c1"
    assert result.retrieved_chunk_ids == ["c1"]
    assert len(result.legal_excerpts) == 1


def test_rejects_answer_with_unsupported_numeric_claim():
    retrieved = [_retrieved("c1")]
    reranked = [_reranked("c1", text="The appeal period is thirty days.")]

    retriever = MagicMock()
    retriever.search.return_value = retrieved
    reranker = MagicMock()
    reranker.rerank.return_value = reranked
    generator = MagicMock()
    # The model invents a number (45) that is not in the query or evidence.
    generator.generate.return_value = "The appeal period is 45 days."

    pipeline = _pipeline(retriever=retriever, reranker=reranker, generator=generator)

    result = pipeline.answer("How long is the appeal period?", language="en")

    assert "could not be validated" in result.answer_text
    assert "45" in result.answer_text


def test_returns_no_evidence_when_reranker_returns_nothing():
    retrieved = [_retrieved("c1")]

    retriever = MagicMock()
    retriever.search.return_value = retrieved
    reranker = MagicMock()
    reranker.rerank.return_value = []

    pipeline = _pipeline(retriever=retriever, reranker=reranker)

    result = pipeline.answer("q", language="en")

    assert "No relevant legal excerpts" in result.answer_text
    # Even with no evidence surviving reranking, retrieved ids are preserved.
    assert result.retrieved_chunk_ids == ["c1"]


def test_generator_receives_the_built_prompt():
    retrieved = [_retrieved("c1")]
    reranked = [_reranked("c1")]

    retriever = MagicMock()
    retriever.search.return_value = retrieved
    reranker = MagicMock()
    reranker.rerank.return_value = reranked
    generator = MagicMock()
    generator.generate.return_value = "An answer."

    pipeline = _pipeline(retriever=retriever, reranker=reranker, generator=generator)
    pipeline.answer("my question", language="en")

    prompt_arg = generator.generate.call_args[0][0]
    assert "my question" in prompt_arg


# -------------------------------------------------------- _diversify_candidates


def test_diversify_keeps_at_most_max_per_document():
    chunks = [_retrieved(f"c{i}", document_id="doc-1") for i in range(6)]

    result = _diversify_candidates(chunks, max_per_document=4)

    assert len(result) == 4


def test_diversify_preserves_original_order():
    chunks = [
        _retrieved("c1", document_id="doc-a"),
        _retrieved("c2", document_id="doc-b"),
        _retrieved("c3", document_id="doc-a"),
    ]

    result = _diversify_candidates(chunks, max_per_document=4)

    assert [chunk.chunk_id for chunk in result] == ["c1", "c2", "c3"]


def test_diversify_allows_documents_up_to_the_limit_independently():
    chunks = [
        _retrieved("a1", document_id="doc-a"),
        _retrieved("a2", document_id="doc-a"),
        _retrieved("b1", document_id="doc-b"),
    ]

    result = _diversify_candidates(chunks, max_per_document=1)

    assert [chunk.chunk_id for chunk in result] == ["a1", "b1"]


def test_diversify_empty_input_returns_empty_list():
    assert _diversify_candidates([], max_per_document=4) == []
