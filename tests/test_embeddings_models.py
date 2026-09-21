"""Tests for embedding configuration."""

from legal_rag.embeddings.models import EmbeddingConfig


def test_embedding_config_defaults() -> None:
    """Use the expected local E5 embedding defaults."""
    config = EmbeddingConfig()

    assert config.model_name == "intfloat/multilingual-e5-base"
    assert config.device == "cpu"
    assert config.normalize_embeddings is True
