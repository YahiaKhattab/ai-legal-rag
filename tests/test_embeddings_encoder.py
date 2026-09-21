"""Tests for the local embedding encoder."""

import numpy as np
import pytest

from legal_rag.embeddings.encoder import EmbeddingEncoder


@pytest.fixture(scope="module")
def encoder() -> EmbeddingEncoder:
    """Create one local encoder for the test module."""
    return EmbeddingEncoder()


def test_embedding_dimension(encoder: EmbeddingEncoder) -> None:
    """Use the expected 768-dimensional E5 embeddings."""
    assert encoder.dimension == 768


def test_document_embedding_is_normalized(
    encoder: EmbeddingEncoder,
) -> None:
    """Return a normalized vector for a document passage."""
    embedding = encoder.encode_document("يشترط لصحة العقد توافر الرضا والمحل والسبب.")

    assert embedding.shape == (768,)
    assert embedding.dtype == np.float32
    assert np.isclose(np.linalg.norm(embedding), 1.0, atol=1e-5)


def test_query_embedding_is_normalized(
    encoder: EmbeddingEncoder,
) -> None:
    """Return a normalized vector for a search query."""
    embedding = encoder.encode_query("ما هي شروط صحة العقد؟")

    assert embedding.shape == (768,)
    assert embedding.dtype == np.float32
    assert np.isclose(np.linalg.norm(embedding), 1.0, atol=1e-5)


def test_document_batch_embedding(
    encoder: EmbeddingEncoder,
) -> None:
    """Encode multiple document passages into one batch."""
    embeddings = encoder.encode_documents(
        [
            "يشترط لصحة العقد توافر الرضا.",
            "يجب أن يكون محل العقد مشروعًا.",
        ]
    )

    assert embeddings.shape == (2, 768)
    assert embeddings.dtype == np.float32


def test_empty_document_is_rejected(
    encoder: EmbeddingEncoder,
) -> None:
    """Reject an empty document."""
    with pytest.raises(
        ValueError,
        match=r"Document text cannot be empty\.",
    ):
        encoder.encode_document("   ")


def test_empty_query_is_rejected(
    encoder: EmbeddingEncoder,
) -> None:
    """Reject an empty query."""
    with pytest.raises(
        ValueError,
        match=r"Query text cannot be empty\.",
    ):
        encoder.encode_query("   ")


def test_empty_document_batch(
    encoder: EmbeddingEncoder,
) -> None:
    """Return an empty batch with the expected embedding dimension."""
    embeddings = encoder.encode_documents([])

    assert embeddings.shape == (0, 768)
    assert embeddings.dtype == np.float32
