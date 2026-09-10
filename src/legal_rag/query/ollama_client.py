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
        max_tokens: int | None = None,
    ) -> str:
        """Generate a non-streaming response using Ollama's chat API."""

        messages: list[dict[str, str]] = []

        if system is not None:
            messages.append(
                {
                    "role": "system",
                    "content": system,
                }
            )

        messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        options: dict[str, object] = {
            "temperature": temperature,
        }

        if max_tokens is not None:
            options["num_predict"] = max_tokens

        payload: dict[str, object] = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": options,
        }

        if format_schema is not None:
            payload["format"] = dict(format_schema)

        response = httpx.post(
            f"{self._base_url}/api/chat",
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
            print(f"Max tokens: {max_tokens}")
            print(f"Response: {response.text}")
            print("==================================\n")

        response.raise_for_status()

        data = response.json()

        message = data.get("message")

        if not isinstance(message, dict):
            raise ValueError("Ollama response did not contain a message")

        generated = message.get("content")

        if not isinstance(generated, str):
            raise ValueError(
                "Ollama response message did not contain generated content"
            )

        return self._clean_thinking(generated)

    @staticmethod
    def _clean_thinking(text: str) -> str:
        """Remove any residual Qwen thinking content from generated text."""

        text = text.strip()

        if "</think>" in text:
            text = text.split("</think>", 1)[1]

        if "<think>" in text:
            text = text.split("<think>", 1)[0]

        return text.strip()