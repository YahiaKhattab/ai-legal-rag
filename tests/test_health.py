from collections.abc import Callable

import httpx

from legal_rag.config import Settings
from legal_rag.health import HealthStatus, check_ollama, check_qdrant


def make_client(
    handler: Callable[[httpx.Request], httpx.Response],
) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_qdrant_is_healthy_when_ready() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="all shards are ready", request=request)

    with make_client(handler) as client:
        result = check_qdrant(client, Settings())

    assert result.status is HealthStatus.HEALTHY
    assert result.service == "qdrant"


def test_qdrant_is_unhealthy_on_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, request=request)

    with make_client(handler) as client:
        result = check_qdrant(client, Settings())

    assert result.status is HealthStatus.UNHEALTHY


def test_ollama_requires_configured_model() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"models": [{"name": "another-model:latest"}]},
            request=request,
        )

    with make_client(handler) as client:
        result = check_ollama(client, Settings())

    assert result.status is HealthStatus.UNHEALTHY
    assert "qwen2.5:3b" in result.detail