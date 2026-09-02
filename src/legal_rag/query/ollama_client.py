from __future__ import annotations

from collections.abc import Mapping

import httpx


class OllamaGenerationClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen3:4b",
        timeout_seconds: float = 200.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout_seconds

    def generate(
        self,
        prompt: str,
        temperature: float = 0.1,
        *,
        system: str | None = None,
        format_schema: Mapping[str, object] | None = None,
    ) -> str:
        """Generate a non-streaming response with optional structured output."""

        payload: dict[str, object] = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": {"temperature": temperature},
        }

        if system is not None:
            payload["system"] = system

        if format_schema is not None:
            payload["format"] = dict(format_schema)

        response = httpx.post(
            f"{self._base_url}/api/generate",
            json=payload,
            timeout=self._timeout,
        )

        if response.status_code >= 400:
            print("\n========== OLLAMA ERROR ==========")
            print(f"Status: {response.status_code}")
            print(f"Model: {self._model}")
            print(f"Prompt characters: {len(prompt)}")
            print(f"System characters: {len(system or '')}")
            print(f"Format schema: {format_schema is not None}")
            print(f"Response: {response.text}")
            print("==================================\n")

        response.raise_for_status()

        generated = response.json().get("response")

        if not isinstance(generated, str):
            raise ValueError("Ollama response did not contain generated text")

        return generated.strip()
