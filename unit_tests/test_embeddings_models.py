"""Unit tests for legal_rag.embeddings.models.EmbeddingConfig (a plain
configuration dataclass)."""

from __future__ import annotations

from legal_rag.embeddings.models import EmbeddingConfig


def test_default_values():
    config = EmbeddingConfig()
    assert config.model_name == "intfloat/multilingual-e5-base"
    assert config.device == "cpu"
    assert config.normalize_embeddings is True


def test_custom_values_are_kept_as_given():
    config = EmbeddingConfig(model_name="custom/model", device="cuda", normalize_embeddings=False)
    assert config.model_name == "custom/model"
    assert config.device == "cuda"
    assert config.normalize_embeddings is False


def test_config_is_frozen():
    config = EmbeddingConfig()
    try:
        config.device = "cuda"  # type: ignore[misc]
        assert False, "Expected a FrozenInstanceError"
    except Exception as error:
        assert "frozen" in str(error).lower() or type(error).__name__ == "FrozenInstanceError"
