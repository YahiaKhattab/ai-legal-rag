import pytest

from legal_rag.config import Settings


def test_settings_normalize_service_urls() -> None:
    settings = Settings(
        qdrant_url="http://localhost:6333/",
        ollama_url="http://localhost:11434/",
    )

    assert settings.qdrant_url == "http://localhost:6333"
    assert settings.ollama_url == "http://localhost:11434"


def test_settings_load_query_pipeline_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LEGAL_RAG_QDRANT_COLLECTION", "legal_chunks_test")
    monkeypatch.setenv("LEGAL_RAG_RETRIEVAL_TOP_K", "12")
    monkeypatch.setenv("LEGAL_RAG_RERANK_TOP_N", "5")
    monkeypatch.setenv("LEGAL_RAG_EVIDENCE_TOP_N", "2")
    monkeypatch.setenv("LEGAL_RAG_EXPERIMENTAL_MIN_DENSE_SCORE", "0.81")

    settings = Settings()

    assert settings.qdrant_collection == "legal_chunks_test"
    assert settings.retrieval_top_k == 12
    assert settings.rerank_top_n == 5
    assert settings.evidence_top_n == 2
    assert settings.experimental_min_dense_score == 0.81
