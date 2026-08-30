"""Unit tests for legal_rag.query.prompt_builder.build_grounded_prompt.

This is pure string-building logic (no network, no model calls), so it
is tested directly with plain RerankedChunk objects.
"""

from __future__ import annotations

from legal_rag.query.models import RerankedChunk
from legal_rag.query.prompt_builder import build_grounded_prompt


def _chunk(**overrides) -> RerankedChunk:
    defaults = dict(
        chunk_id="c1",
        document_id="d1",
        score=0.9,
        text="Some legal excerpt text.",
        source_file="law.pdf",
        section_type="article",
        section_title="Article 5",
        page=3,
        language="en",
        document_type="statute",
        source="test",
        rerank_score=0.8,
    )
    defaults.update(overrides)
    return RerankedChunk(**defaults)


def test_builds_one_citation_per_chunk_in_order():
    chunks = [_chunk(chunk_id="c1"), _chunk(chunk_id="c2")]
    prompt, citations = build_grounded_prompt("What is the rule?", chunks, language="en")

    assert [citation.marker for citation in citations] == ["[1]", "[2]"]
    assert [citation.chunk_id for citation in citations] == ["c1", "c2"]


def test_citation_carries_source_file_page_and_section():
    chunks = [_chunk(source_file="civil_code.pdf", page=10, section_title="Article 9")]
    _, citations = build_grounded_prompt("q", chunks)

    citation = citations[0]
    assert citation.source_file == "civil_code.pdf"
    assert citation.page == 10
    assert citation.section_title == "Article 9"


def test_prompt_includes_the_query_text():
    chunks = [_chunk()]
    prompt, _ = build_grounded_prompt("How long is the appeal period?", chunks)
    assert "How long is the appeal period?" in prompt


def test_prompt_includes_evidence_text_and_marker():
    chunks = [_chunk(text="The appeal period is thirty days.")]
    prompt, _ = build_grounded_prompt("q", chunks)
    assert "[1]" in prompt
    assert "The appeal period is thirty days." in prompt


def test_prompt_uses_placeholder_when_no_chunks_given():
    prompt, citations = build_grounded_prompt("q", [])
    assert "(no relevant excerpts found)" in prompt
    assert citations == []


def test_arabic_language_selects_arabic_system_prompt():
    prompt, _ = build_grounded_prompt("سؤال", [_chunk()], language="ar")
    assert "مساعد قانوني" in prompt


def test_english_language_selects_english_system_prompt():
    prompt, _ = build_grounded_prompt("question", [_chunk()], language="en")
    assert "You are a legal assistant." in prompt


def test_unknown_language_falls_back_to_mixed_prompt():
    prompt, _ = build_grounded_prompt("q", [_chunk()], language="fr")
    assert "Respond in the same language as the user's question." in prompt


def test_locator_omits_page_when_not_available():
    chunks = [_chunk(section_title=None, page=None, source_file="doc.pdf")]
    prompt, _ = build_grounded_prompt("q", chunks)
    # No dangling "page:" or "section:" text should appear when both are absent.
    assert "page:" not in prompt
    assert "section:" not in prompt


def test_source_falls_back_to_document_id_when_no_source_file():
    chunks = [_chunk(source_file=None, document_id="doc-42")]
    prompt, _ = build_grounded_prompt("q", chunks)
    assert "doc-42" in prompt
