"""Unit tests for legal_rag.query.reranker.CrossEncoderReranker.

CrossEncoder (from sentence_transformers) is patched with a fake that
returns scores we control, so we can prove the reranker sorts
descending and truncates to top_n without needing the real ML model.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from legal_rag.query.models import RetrievedChunk
from legal_rag.query.reranker import CrossEncoderReranker, get_default_reranker


def _candidate(chunk_id, text):
    return RetrievedChunk(chunk_id=chunk_id, document_id="d1", score=0.5, text=text)


class _FakeCrossEncoder:
    def __init__(self, model_name, device, max_length):
        self.model_name = model_name

    def predict(self, pairs):
        # Score is simply the length of the candidate text -- longer
        # "more relevant" text wins, giving predictable ordering.
        return [len(text) for _query, text in pairs]


@pytest.fixture
def reranker():
    with patch("legal_rag.query.reranker.CrossEncoder", _FakeCrossEncoder):
        yield CrossEncoderReranker()


def test_rerank_sorts_candidates_by_score_descending(reranker):
    candidates = [_candidate("short", "a"), _candidate("long", "aaaaaaaaaa"), _candidate("mid", "aaa")]

    result = reranker.rerank("query", candidates, top_n=3)

    assert [chunk.chunk_id for chunk in result] == ["long", "mid", "short"]


def test_rerank_truncates_to_top_n(reranker):
    candidates = [_candidate(str(i), "a" * i) for i in range(1, 6)]

    result = reranker.rerank("query", candidates, top_n=2)

    assert len(result) == 2
    assert result[0].chunk_id == "5"
    assert result[1].chunk_id == "4"


def test_rerank_returns_empty_list_for_no_candidates(reranker):
    assert reranker.rerank("query", [], top_n=6) == []


def test_rerank_preserves_original_chunk_fields_and_adds_rerank_score(reranker):
    candidate = RetrievedChunk(
        chunk_id="c1", document_id="d1", score=0.9, text="hello", source_file="doc.pdf"
    )

    result = reranker.rerank("query", [candidate], top_n=1)

    assert result[0].chunk_id == "c1"
    assert result[0].source_file == "doc.pdf"
    assert result[0].rerank_score == pytest.approx(5.0)  # len("hello")


def test_get_default_reranker_is_a_singleton():
    get_default_reranker.cache_clear()
    with patch("legal_rag.query.reranker.CrossEncoder", _FakeCrossEncoder):
        first = get_default_reranker()
        second = get_default_reranker()
    assert first is second
    get_default_reranker.cache_clear()
