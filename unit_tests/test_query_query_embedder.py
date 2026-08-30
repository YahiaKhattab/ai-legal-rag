"""Unit tests for legal_rag.query.query_embedder.QueryEmbedder.

Same approach as test_embeddings_encoder.py: SentenceTransformer is
patched with a small fake so no real model download/torch is required.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from legal_rag.query.query_embedder import QueryEmbedder, get_default_query_embedder


class _FakeSentenceTransformer:
    def __init__(self, model_name, device):
        self.model_name = model_name
        self.device = device

    def encode(self, text, normalize_embeddings, convert_to_numpy):
        # Deterministic vector based on prefixed-text length so tests can
        # assert the "query: " prefix was actually applied.
        return np.array([float(len(text))] * 3)


@pytest.fixture
def embedder():
    with patch("legal_rag.query.query_embedder.SentenceTransformer", _FakeSentenceTransformer):
        yield QueryEmbedder()


def test_encode_query_adds_query_prefix_and_strips_whitespace(embedder):
    vector = embedder.encode_query("  hello  ")
    # "query: hello" == 12 characters.
    assert vector[0] == pytest.approx(12.0)
    assert isinstance(vector, list)


def test_encode_query_returns_float32_compatible_values(embedder):
    vector = embedder.encode_query("hi")
    assert all(isinstance(value, float) for value in vector)


def test_encode_query_rejects_empty_string(embedder):
    with pytest.raises(ValueError, match="must not be empty"):
        embedder.encode_query("")


def test_encode_query_rejects_whitespace_only_string(embedder):
    with pytest.raises(ValueError, match="must not be empty"):
        embedder.encode_query("    ")


def test_get_default_query_embedder_is_a_process_wide_singleton(monkeypatch):
    get_default_query_embedder.cache_clear()
    with patch("legal_rag.query.query_embedder.SentenceTransformer", _FakeSentenceTransformer):
        first = get_default_query_embedder()
        second = get_default_query_embedder()

    assert first is second
    get_default_query_embedder.cache_clear()
