"""Independent telemetry settings; never serialize application secrets."""

from urllib.parse import urlsplit

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ObservabilitySettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="LEGAL_RAG_PHOENIX_", extra="ignore"
    )

    enabled: bool = False
    endpoint: str = "http://127.0.0.1:6006/v1/traces"
    project: str = Field(default="legal-rag-development", min_length=1, max_length=100)

    @field_validator("endpoint")
    @classmethod
    def validate_endpoint(cls, value: str) -> str:
        parsed = urlsplit(value)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or parsed.path != "/v1/traces"
        ):
            raise ValueError(
                "use an HTTP(S) OTLP endpoint ending in /v1/traces without credentials"
            )
        return value
