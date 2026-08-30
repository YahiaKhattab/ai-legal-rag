"""Unit tests for legal_rag.health (Qdrant/Ollama health checks).

Approach
--------
health.py talks to two real network services (Qdrant, Ollama) through
httpx.Client. A unit test must NOT make real network calls, so every test
here builds a fake httpx.Client-like object whose `.get()` returns a
pre-built response, or raises an httpx error, exactly as the real client
would.
"""

from __future__ import annotations

import httpx
import pytest

from legal_rag.config import Settings
from legal_rag.health import (
    HealthStatus,
    check_ollama,
    check_qdrant,
    run_health_checks,
)


def _settings() -> Settings:
    return Settings(_env_file=None)


class _FakeResponse:
    def __init__(self, *, json_data=None, status_code=200):
        self._json_data = json_data
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("GET", "http://fake")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError("error", request=request, response=response)

    def json(self):
        return self._json_data


class _FakeClient:
    """Stands in for httpx.Client; returns queued responses in call order."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.requested_urls = []

    def get(self, url):
        self.requested_urls.append(url)
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def test_check_qdrant_healthy_when_readyz_succeeds():
    client = _FakeClient([_FakeResponse(status_code=200)])

    result = check_qdrant(client, _settings())

    assert result.service == "qdrant"
    assert result.status is HealthStatus.HEALTHY
    assert result.detail == "ready"
    assert client.requested_urls == ["http://localhost:6333/readyz"]


def test_check_qdrant_unhealthy_on_http_error():
    client = _FakeClient([httpx.ConnectError("refused")])

    result = check_qdrant(client, _settings())

    assert result.status is HealthStatus.UNHEALTHY
    assert "ConnectError" in result.detail


def test_check_qdrant_unhealthy_on_bad_status_code():
    client = _FakeClient([_FakeResponse(status_code=500)])

    result = check_qdrant(client, _settings())

    assert result.status is HealthStatus.UNHEALTHY


def test_check_ollama_healthy_when_model_installed():
    settings = Settings(_env_file=None, generation_model="qwen2.5:3b")
    client = _FakeClient(
        [_FakeResponse(json_data={"models": [{"name": "qwen2.5:3b", "model": "qwen2.5:3b"}]})]
    )

    result = check_ollama(client, settings)

    assert result.status is HealthStatus.HEALTHY
    assert "qwen2.5:3b" in result.detail


def test_check_ollama_unhealthy_when_model_missing():
    settings = Settings(_env_file=None, generation_model="qwen2.5:3b")
    client = _FakeClient([_FakeResponse(json_data={"models": [{"name": "llama3:8b"}]})])

    result = check_ollama(client, settings)

    assert result.status is HealthStatus.UNHEALTHY
    assert "not installed" in result.detail


def test_check_ollama_unhealthy_on_malformed_json():
    settings = Settings(_env_file=None)
    client = _FakeClient([_FakeResponse(json_data={"unexpected": "shape"})])

    result = check_ollama(client, settings)

    assert result.status is HealthStatus.UNHEALTHY


def test_check_ollama_unhealthy_on_connection_error():
    client = _FakeClient([httpx.ConnectError("refused")])

    result = check_ollama(client, _settings())

    assert result.status is HealthStatus.UNHEALTHY


def test_run_health_checks_uses_a_real_httpx_client(monkeypatch):
    """run_health_checks() builds its own httpx.Client internally.

    We patch httpx.Client itself (not check_qdrant/check_ollama) so this
    test also proves run_health_checks wires the two checks together
    and returns both results in order.
    """

    class _Recorder:
        def __init__(self, timeout):
            self.timeout = timeout

        def __enter__(self):
            return _FakeClient(
                [
                    _FakeResponse(status_code=200),
                    _FakeResponse(json_data={"models": [{"name": "qwen2.5:3b"}]}),
                ]
            )

        def __exit__(self, *exc_info):
            return False

    monkeypatch.setattr(httpx, "Client", _Recorder)

    settings = Settings(_env_file=None, generation_model="qwen2.5:3b")
    results = run_health_checks(settings)

    assert [result.service for result in results] == ["qdrant", "ollama"]
    assert all(result.status is HealthStatus.HEALTHY for result in results)
