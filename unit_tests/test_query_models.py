"""Unit tests for legal_rag.query.models (plain data contracts).

These dataclasses have no custom logic, so the tests focus on the two
things that can actually break silently: default values, and that
RerankedChunk really is a RetrievedChunk (inheritance used elsewhere in
the pipeline, e.g. `RerankedChunk(**candidate.__dict__, ...)`).
"""

from __future__ import annotations

from legal_rag.query.models import (
    Citation,
    CitedAnswer,
    LegalExcerpt,
    RerankedChunk,
    RetrievedChunk,
)


def test_retrieved_chunk_defaults():
    chunk = RetrievedChunk(chunk_id="c1", document_id="d1", score=0.9, text="hello")
    assert chunk.source_file is None
    assert chunk.payload == {}


def test_reranked_chunk_is_a_retrieved_chunk_subclass():
    reranked = RerankedChunk(chunk_id="c1", document_id="d1", score=0.9, text="hi", rerank_score=0.5)
    assert isinstance(reranked, RetrievedChunk)
    assert reranked.rerank_score == 0.5


def test_reranked_chunk_default_rerank_score_is_zero():
    reranked = RerankedChunk(chunk_id="c1", document_id="d1", score=0.9, text="hi")
    assert reranked.rerank_score == 0.0


def test_citation_holds_all_locator_fields():
    citation = Citation(
        marker="[1]",
        chunk_id="c1",
        document_id="d1",
        source_file="doc.pdf",
        section_title="Article 5",
        page=3,
    )
    assert citation.marker == "[1]"
    assert citation.page == 3


def test_legal_excerpt_fields():
    excerpt = LegalExcerpt(
        marker="[1]", text="text", source_file="doc.pdf", section_title=None, page=1, chunk_id="c1"
    )
    assert excerpt.text == "text"


def test_cited_answer_defaults_legal_excerpts_to_empty_list():
    answer = CitedAnswer(
        query="q",
        answer_text="a",
        language="en",
        citations=[],
        retrieved_chunk_ids=[],
    )
    assert answer.legal_excerpts == []


def test_cited_answer_two_instances_get_independent_default_lists():
    # Regression guard against the classic mutable-default-argument bug.
    answer_a = CitedAnswer(query="q1", answer_text="a", language="en", citations=[], retrieved_chunk_ids=[])
    answer_b = CitedAnswer(query="q2", answer_text="a", language="en", citations=[], retrieved_chunk_ids=[])

    answer_a.legal_excerpts.append(
        LegalExcerpt(marker="[1]", text="t", source_file=None, section_title=None, page=None, chunk_id="c1")
    )

    assert answer_a.legal_excerpts != answer_b.legal_excerpts
    assert len(answer_b.legal_excerpts) == 0
