from legal_rag.config import Settings


def test_settings_normalize_service_urls() -> None:
    settings = Settings(
        qdrant_url="http://localhost:6333/",
        ollama_url="http://localhost:11434/",
    )

    assert settings.qdrant_url == "http://localhost:6333"
    assert settings.ollama_url == "http://localhost:11434"