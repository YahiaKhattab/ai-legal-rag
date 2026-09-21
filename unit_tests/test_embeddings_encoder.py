"""Unit tests for legal_rag.embeddings.encoder.EmbeddingEncoder.

The real SentenceTransformer model requires downloading multi-hundred-MB
weights and torch, so we patch it with a small fake model exposing only
the two methods EmbeddingEncoder actually calls:
`get_sentence_embedding_dimension()` and `encode(...)`.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from legal_rag.embeddings.encoder import EmbeddingEncoder
from legal_rag.embeddings.models import EmbeddingConfig


class _FakeSentenceTransformer:
    def __init__(self, model_name, device):
        self.model_name = model_name
        self.device = device
        self.encode_calls = []

    def get_sentence_embedding_dimension(self):
        return 4

    def encode(self, texts, normalize_embeddings, convert_to_numpy, show_progress_bar):
        self.encode_calls.append(texts)
        # One deterministic 4-dim vector per input text.
        return np.array([[float(len(text))] * 4 for text in texts], dtype=np.float32)


@pytest.fixture
def encoder():
    with patch(
        "legal_rag.embeddings.encoder.SentenceTransformer", _FakeSentenceTransformer
    ):
        yield EmbeddingEncoder(EmbeddingConfig(model_name="fake-model"))


def test_dimension_property_reads_from_the_model(encoder):
    assert encoder.dimension == 4


def test_encode_document_adds_passage_prefix(encoder):
    vector = encoder.encode_document("hello")
    assert vector.shape == (4,)
    assert vector.dtype == np.float32
    # "passage: hello" is 14 characters -> our fake model encodes length.
    assert vector[0] == pytest.approx(14.0)


def test_encode_query_adds_query_prefix(encoder):
    vector = encoder.encode_query("hello")
    # "query: hello" is 12 characters.
    assert vector[0] == pytest.approx(12.0)


def test_encode_document_rejects_empty_text(encoder):
    with pytest.raises(ValueError, match="Document text cannot be empty"):
        encoder.encode_document("   ")


def test_encode_query_rejects_empty_text(encoder):
    with pytest.raises(ValueError, match="Query text cannot be empty"):
        encoder.encode_query("")


def test_encode_documents_batch_preserves_order(encoder):
    vectors = encoder.encode_documents(["a", "bb", "ccc"])
    assert vectors.shape == (3, 4)
    # Lengths differ (with the "passage: " prefix added) so order is provable.
    assert vectors[0][0] < vectors[1][0] < vectors[2][0]


def test_encode_documents_empty_list_returns_empty_array_with_correct_dimension(encoder):
    vectors = encoder.encode_documents([])
    assert vectors.shape == (0, 4)


def test_encode_documents_rejects_any_empty_text_in_the_batch(encoder):
    with pytest.raises(ValueError, match="Document text cannot be empty"):
        encoder.encode_documents(["valid", "   "])
