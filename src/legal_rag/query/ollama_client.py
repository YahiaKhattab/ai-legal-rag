"""Thin client for the already-configured local Ollama generator.

Reuses LEGAL_RAG_OLLAMA_URL / LEGAL_RAG_GENERATION_MODEL, which are already
defined in config.py and health-checked in health.py -- this module is what
finally *uses* that connection for RAG generation instead of just probing
it.
"""
from __future__ import annotations

import httpx


class OllamaGenerationClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5:3b",
        timeout_seconds: float = 60.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout_seconds

    def generate(self, prompt: str, temperature: float = 0.1) -> str:
        """Non-streaming generation. Low temperature by default since this
        is grounded legal summarization, not creative writing -- we want
        the model to stick closely to the provided excerpts.
        """
        response = httpx.post(
            f"{self._base_url}/api/generate",
            json={
                "model": self._model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature},
            },
            timeout=self._timeout,
        )
        response.raise_for_status()
        return response.json()["response"].strip()
