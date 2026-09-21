"""Unit tests for legal_rag.query.retriever.LegalRetriever.

The retriever talks to Qdrant (via QdrantVectorStore) and to a
QueryEmbedder. Both are injected as fakes/mocks -- there is no network
or model I/O in this test file at all.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from legal_rag.query.retriever import LegalRetriever, RetrievalFilters


def _fake_point(chunk_id, document_id, score, payload_overrides=None):
    point = MagicMock()
    point.id = chunk_id
    point.score = score
    payload = {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "original_text": "the legal text",
        "source_file": "doc.pdf",
        "section_type": "article",
        "section_title": "Article 5",
        "page_start": 3,
        "language": "ar",
        "document_type": "statute",
        "source": "tests",
    }
    if payload_overrides:
        payload.update(payload_overrides)
    point.payload = payload
    return point


def _fake_store_with_results(points):
    store = MagicMock()
    store.collection_name = "legal_chunks"
    store.client.query_points.return_value = MagicMock(points=points)
    return store


def _fake_embedder(vector=None):
    embedder = MagicMock()
    embedder.encode_query.return_value = vector or [0.1, 0.2, 0.3]
    return embedder


def test_search_embeds_the_query_and_returns_retrieved_chunks():
    store = _fake_store_with_results([_fake_point("c1", "d1", 0.9)])
    embedder = _fake_embedder([0.1, 0.2])

    retriever = LegalRetriever(store=store, embedder=embedder)
    results = retriever.search("what is the rule?", top_k=5)

    embedder.encode_query.assert_called_once_with("what is the rule?")
    assert len(results) == 1
    assert results[0].chunk_id == "c1"
    assert results[0].score == pytest.approx(0.9)
    assert results[0].text == "the legal text"
    assert results[0].page == 3


def test_search_passes_top_k_and_vector_to_qdrant():
    store = _fake_store_with_results([])
    embedder = _fake_embedder([0.5, 0.5])

    LegalRetriever(store=store, embedder=embedder).search("q", top_k=7)

    _, kwargs = store.client.query_points.call_args
    assert kwargs["limit"] == 7
    assert kwargs["query"] == [0.5, 0.5]
    assert kwargs["collection_name"] == "legal_chunks"


def test_search_returns_empty_list_when_qdrant_has_no_hits():
    store = _fake_store_with_results([])
    embedder = _fake_embedder()

    results = LegalRetriever(store=store, embedder=embedder).search("q")

    assert results == []


def test_search_falls_back_to_text_store_when_payload_has_no_text():
    point = _fake_point("c1", "d1", 0.7, payload_overrides={"original_text": None, "normalized_text": None})
    store = _fake_store_with_results([point])
    embedder = _fake_embedder()
    text_store = MagicMock()
    text_store.get_text.return_value = "text from the fallback store"

    retriever = LegalRetriever(store=store, embedder=embedder, text_store=text_store)
    results = retriever.search("q")

    text_store.get_text.assert_called_once_with("c1")
    assert results[0].text == "text from the fallback store"


def test_search_without_filters_passes_no_qdrant_filter():
    store = _fake_store_with_results([])
    embedder = _fake_embedder()

    LegalRetriever(store=store, embedder=embedder).search("q")

    _, kwargs = store.client.query_points.call_args
    assert kwargs["query_filter"] is None


def test_search_with_filters_builds_a_qdrant_filter():
    store = _fake_store_with_results([])
    embedder = _fake_embedder()
    filters = RetrievalFilters(language="ar", document_type="statute")

    LegalRetriever(store=store, embedder=embedder).search("q", filters=filters)

    _, kwargs = store.client.query_points.call_args
    assert kwargs["query_filter"] is not None
    # Two conditions (language + document_type) should be present.
    assert len(kwargs["query_filter"].must) == 2


def test_search_with_no_filter_fields_set_returns_none_filter():
    store = _fake_store_with_results([])
    embedder = _fake_embedder()
    empty_filters = RetrievalFilters()

    LegalRetriever(store=store, embedder=embedder).search("q", filters=empty_filters)

    _, kwargs = store.client.query_points.call_args
    assert kwargs["query_filter"] is None


def test_search_passes_score_threshold_through():
    store = _fake_store_with_results([])
    embedder = _fake_embedder()

    LegalRetriever(store=store, embedder=embedder).search("q", score_threshold=0.5)

    _, kwargs = store.client.query_points.call_args
    assert kwargs["score_threshold"] == 0.5
