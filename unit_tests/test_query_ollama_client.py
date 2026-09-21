"""Unit tests for legal_rag.query.ollama_client.OllamaGenerationClient.

Patches httpx.post so no real Ollama server is required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from legal_rag.query.ollama_client import OllamaGenerationClient


def _fake_response(text: str, status_code: int = 200):
    response = MagicMock()
    response.json.return_value = {"response": text}
    response.raise_for_status = MagicMock()
    if status_code >= 400:
        request = httpx.Request("POST", "http://fake")
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=request, response=httpx.Response(status_code, request=request)
        )
    return response


def test_generate_sends_expected_payload_and_returns_stripped_text():
    with patch("legal_rag.query.ollama_client.httpx.post") as fake_post:
        fake_post.return_value = _fake_response("  the answer  ")

        client = OllamaGenerationClient(base_url="http://localhost:11434", model="qwen2.5:3b")
        result = client.generate("What is the rule?", temperature=0.2)

        assert result == "the answer"
        _, kwargs = fake_post.call_args
        assert kwargs["json"]["model"] == "qwen2.5:3b"
        assert kwargs["json"]["prompt"] == "What is the rule?"
        assert kwargs["json"]["stream"] is False
        assert kwargs["json"]["options"]["temperature"] == 0.2


def test_base_url_trailing_slash_is_removed():
    with patch("legal_rag.query.ollama_client.httpx.post") as fake_post:
        fake_post.return_value = _fake_response("answer")
        client = OllamaGenerationClient(base_url="http://localhost:11434/")
        client.generate("q")

        args, _ = fake_post.call_args
        assert args[0] == "http://localhost:11434/api/generate"


def test_generate_uses_default_temperature_when_not_specified():
    with patch("legal_rag.query.ollama_client.httpx.post") as fake_post:
        fake_post.return_value = _fake_response("answer")
        client = OllamaGenerationClient()
        client.generate("q")

        _, kwargs = fake_post.call_args
        assert kwargs["json"]["options"]["temperature"] == 0.1


def test_generate_raises_on_http_error_status():
    with patch("legal_rag.query.ollama_client.httpx.post") as fake_post:
        fake_post.return_value = _fake_response("", status_code=500)
        client = OllamaGenerationClient()

        with pytest.raises(httpx.HTTPStatusError):
            client.generate("q")


def test_generate_passes_configured_timeout():
    with patch("legal_rag.query.ollama_client.httpx.post") as fake_post:
        fake_post.return_value = _fake_response("answer")
        client = OllamaGenerationClient(timeout_seconds=42.0)
        client.generate("q")

        _, kwargs = fake_post.call_args
        assert kwargs["timeout"] == 42.0
