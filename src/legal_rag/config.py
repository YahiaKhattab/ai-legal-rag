from urllib.parse import urlparse

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="LEGAL_RAG_",
        case_sensitive=False,
        extra="ignore",
    )

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "legal_chunks"
    ollama_url: str = "http://localhost:11434"
    generation_model: str = "qwen3:4b"
    health_timeout_seconds: float = Field(default=5.0, gt=0, le=60)

    embedding_model: str = "intfloat/multilingual-e5-base"
    embedding_device: str = "cpu"
    rerank_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    rerank_device: str = "cpu"

    retrieval_top_k: int = Field(default=20, ge=1, le=200)
    rerank_top_n: int = Field(default=6, ge=1, le=50)
    evidence_top_n: int = Field(default=2, ge=1, le=20)

    generation_timeout_seconds: float = Field(default=200.0, gt=0, le=600)
    generation_temperature: float = Field(default=0.1, ge=0, le=2)
    generation_retry_count: int = Field(default=1, ge=0, le=3)
    maximum_context_characters: int = Field(default=12_000, ge=1_000, le=100_000)

    evidence_sufficiency_enabled: bool = True
    experimental_min_dense_score: float = Field(default=0.82, ge=-1, le=1)
    experimental_identifier_override_score: float = Field(default=0.75, ge=-1, le=1)
    experimental_min_rerank_score: float | None = -1.0
    experimental_max_dense_score_drop: float = Field(default=0.02, ge=0, le=2)

    @field_validator("qdrant_url", "ollama_url")
    @classmethod
    def validate_service_url(cls, value: str) -> str:
        normalized = value.rstrip("/")
        parsed = urlparse(normalized)

        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("must be a valid HTTP or HTTPS URL")

        return normalized

    @model_validator(mode="after")
    def validate_pipeline_limits(self) -> "Settings":
        if self.rerank_top_n > self.retrieval_top_k:
            raise ValueError("rerank_top_n must not exceed retrieval_top_k")
        if self.evidence_top_n > self.rerank_top_n:
            raise ValueError("evidence_top_n must not exceed rerank_top_n")
        if self.experimental_identifier_override_score > self.experimental_min_dense_score:
            raise ValueError(
                "experimental_identifier_override_score must not exceed "
                "experimental_min_dense_score"
            )
        return self
