from urllib.parse import urlparse

from pydantic import Field, field_validator
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
    ollama_url: str = "http://localhost:11434"
    generation_model: str = "qwen2.5:3b"
    health_timeout_seconds: float = Field(default=5.0, gt=0, le=60)

    @field_validator("qdrant_url", "ollama_url")
    @classmethod
    def validate_service_url(cls, value: str) -> str:
        normalized = value.rstrip("/")
        parsed = urlparse(normalized)

        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("must be a valid HTTP or HTTPS URL")

        return normalized