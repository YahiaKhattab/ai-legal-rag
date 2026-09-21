"""Configuration for local embedding generation."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EmbeddingConfig:
    """Settings used by the local embedding encoder."""

    model_name: str = "intfloat/multilingual-e5-base"
    device: str = "cpu"
    normalize_embeddings: bool = True
