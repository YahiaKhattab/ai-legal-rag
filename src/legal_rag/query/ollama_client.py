from __future__ import annotations

import httpx


class OllamaGenerationClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5:3b",
        timeout_seconds: float = 200.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout_seconds

    def generate(self, prompt: str, temperature: float = 0.1) -> str:
        """Non-streaming generation."""

        response = httpx.post(
            f"{self._base_url}/api/generate",
            json={
                "model": self._model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                },
            },
            timeout=self._timeout,
        )
        response.raise_for_status()
        return response.json()["response"].strip()