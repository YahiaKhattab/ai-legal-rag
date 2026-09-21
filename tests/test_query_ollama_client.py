from typing import Any

import httpx
import pytest

from legal_rag.query.ollama_client import OllamaGenerationClient


def test_ollama_client_sends_system_and_json_schema_separately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_post(
        url: str,
        *,
        json: dict[str, object],
        timeout: float,
    ) -> httpx.Response:
        captured.update({"url": url, "json": json, "timeout": timeout})
        request = httpx.Request("POST", url)
        return httpx.Response(200, json={"response": '{"answer":"ok"}'}, request=request)

    monkeypatch.setattr(httpx, "post", fake_post)
    client = OllamaGenerationClient(
        base_url="http://ollama.local/",
        model="qwen-test",
        timeout_seconds=42,
    )

    result = client.generate(
        "untrusted user data",
        system="trusted system instructions",
        format_schema={"type": "object"},
    )

    assert result == '{"answer":"ok"}'
    assert captured["url"] == "http://ollama.local/api/generate"
    assert captured["timeout"] == 42
    payload = captured["json"]
    assert isinstance(payload, dict)
    assert payload["system"] == "trusted system instructions"
    assert payload["prompt"] == "untrusted user data"
    assert payload["format"] == {"type": "object"}
    assert payload["think"] is False
