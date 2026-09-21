from __future__ import annotations

import time
from collections.abc import Mapping

import httpx

from legal_rag.observability.tracing import attributes, traced


class OllamaGenerationClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "gemma3:4b",
        timeout_seconds: float = 200.0,
        max_tokens: int = 384,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout_seconds
        self._max_tokens = max_tokens

    @traced("generation.attempt", "LLM")
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

        attributes(**{
            "llm.model_name": self._model,
            "rag.structured_output": format_schema is not None,
            "rag.generation_purpose": "answer" if format_schema is not None else "contextualization",
        })
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

        effective_max_tokens = (
            self._max_tokens
            if max_tokens is None
            else max_tokens
        )

        options: dict[str, object] = {
            "temperature": temperature,
            "num_predict": effective_max_tokens,
        }

        payload: dict[str, object] = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": options,
        }

        if format_schema is not None:
            payload["format"] = dict(format_schema)

        print("\n========== OLLAMA GENERATION ==========")
        print(f"[Ollama] Model: {self._model}")
        print(f"[Ollama] Timeout: {self._timeout}s")
        print(
            "[Ollama] Max tokens: "
            f"{effective_max_tokens}"
        )
        print(
            "[Ollama] Prompt characters: "
            f"{len(prompt)}"
        )
        print(
            "[Ollama] System characters: "
            f"{len(system or '')}"
        )
        print(
            "[Ollama] Structured format: "
            f"{format_schema is not None}"
        )

        generation_start = time.perf_counter()

        response = httpx.post(
            f"{self._base_url}/api/chat",
            json=payload,
            timeout=self._timeout,
        )

        generation_time = (
            time.perf_counter() - generation_start
        )

        print(
            "[Ollama] HTTP generation time: "
            f"{generation_time:.3f}s"
        )

        if response.status_code >= 400:
            print("\n========== OLLAMA ERROR ==========")
            print(f"Status: {response.status_code}")
            print(f"Model: {self._model}")
            print(f"Prompt characters: {len(prompt)}")
            print(f"System characters: {len(system or '')}")
            print(
                "Format schema: "
                f"{format_schema is not None}"
            )
            print(
                "Max tokens: "
                f"{effective_max_tokens}"
            )
            print(f"Response: {response.text}")
            print("==================================\n")

        response.raise_for_status()

        data = response.json()
        attributes(**{
            "llm.token_count.prompt": data.get("prompt_eval_count"),
            "llm.token_count.completion": data.get("eval_count"),
        })

        message = data.get("message")

        if not isinstance(message, dict):
            raise ValueError(
                "Ollama response did not contain a message"
            )

        generated = message.get("content")

        if not isinstance(generated, str):
            raise ValueError(
                "Ollama response message did not contain "
                "generated content"
            )

        cleaned = self._clean_thinking(generated)

        print(
            "[Ollama] Generated characters: "
            f"{len(cleaned)}"
        )
        print("=======================================\n")

        return cleaned

    @staticmethod
    def _clean_thinking(text: str) -> str:
        """Remove any residual Qwen thinking content from generated text."""

        text = text.strip()

        if "</think>" in text:
            text = text.split("</think>", 1)[1]

        if "<think>" in text:
            text = text.split("<think>", 1)[0]

        return text.strip()