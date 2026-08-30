"""Unit tests for legal_rag.config.Settings.

What this file covers
----------------------
Settings is a pydantic-settings object that loads configuration from
environment variables (LEGAL_RAG_*). We test:
  - default values when no environment variables are set
  - that environment variables override the defaults
  - the custom validator that rejects malformed qdrant/ollama URLs
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from legal_rag.config import Settings


def test_defaults_are_used_when_no_env_vars_set(monkeypatch):
    # Make sure no leftover LEGAL_RAG_* variables leak in from the shell
    # running the tests and disable the .env file lookup.
    for name in ("LEGAL_RAG_QDRANT_URL", "LEGAL_RAG_OLLAMA_URL", "LEGAL_RAG_GENERATION_MODEL"):
        monkeypatch.delenv(name, raising=False)

    settings = Settings(_env_file=None)

    assert settings.qdrant_url == "http://localhost:6333"
    assert settings.ollama_url == "http://localhost:11434"
    assert settings.generation_model == "qwen2.5:3b"
    assert settings.health_timeout_seconds == 5.0


def test_environment_variables_override_defaults(monkeypatch):
    monkeypatch.setenv("LEGAL_RAG_QDRANT_URL", "http://my-qdrant-host:6333")
    monkeypatch.setenv("LEGAL_RAG_GENERATION_MODEL", "llama3:8b")

    settings = Settings(_env_file=None)

    assert settings.qdrant_url == "http://my-qdrant-host:6333"
    assert settings.generation_model == "llama3:8b"


def test_trailing_slash_is_stripped_from_urls():
    settings = Settings(_env_file=None, qdrant_url="http://localhost:6333/")

    assert settings.qdrant_url == "http://localhost:6333"


@pytest.mark.parametrize(
    "bad_url",
    [
        "not-a-url",
        "ftp://localhost:6333",
        "localhost:6333",
        "",
    ],
)
def test_invalid_service_url_raises(bad_url):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, qdrant_url=bad_url)


def test_health_timeout_must_be_positive_and_bounded():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, health_timeout_seconds=0)

    with pytest.raises(ValidationError):
        Settings(_env_file=None, health_timeout_seconds=61)

    # A valid boundary value should not raise.
    settings = Settings(_env_file=None, health_timeout_seconds=60)
    assert settings.health_timeout_seconds == 60
