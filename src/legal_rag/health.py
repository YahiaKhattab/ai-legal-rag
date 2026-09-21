from dataclasses import dataclass
from enum import StrEnum

import httpx
from pydantic import BaseModel, ValidationError

from legal_rag.config import Settings


class HealthStatus(StrEnum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"


@dataclass(frozen=True, slots=True)
class HealthResult:
    service: str
    status: HealthStatus
    detail: str


class _OllamaModel(BaseModel):
    name: str
    model: str = ""


class _OllamaTagsResponse(BaseModel):
    models: list[_OllamaModel]


def check_qdrant(client: httpx.Client, settings: Settings) -> HealthResult:
    """Check whether Qdrant is ready to accept requests."""

    headers = {"api-key": settings.qdrant_api_key} if settings.qdrant_api_key else {}

    try:
        response = client.get(f"{settings.qdrant_url}/readyz", headers=headers)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        return HealthResult(
            service="qdrant",
            status=HealthStatus.UNHEALTHY,
            detail=f"{type(exc).__name__}: {exc}",
        )

    return HealthResult(
        service="qdrant",
        status=HealthStatus.HEALTHY,
        detail="ready",
    )


def check_ollama(client: httpx.Client, settings: Settings) -> HealthResult:
    """Check Ollama and confirm that the configured model is installed."""

    try:
        response = client.get(f"{settings.ollama_url}/api/tags")
        response.raise_for_status()
        payload = _OllamaTagsResponse.model_validate(response.json())
    except (httpx.HTTPError, ValueError) as exc:
        return HealthResult(
            service="ollama",
            status=HealthStatus.UNHEALTHY,
            detail=f"{type(exc).__name__}: {exc}",
        )

    installed_models = {item.name for item in payload.models}
    installed_models.update(item.model for item in payload.models if item.model)

    if settings.generation_model not in installed_models:
        return HealthResult(
            service="ollama",
            status=HealthStatus.UNHEALTHY,
            detail=f"required model is not installed: {settings.generation_model}",
        )

    return HealthResult(
        service="ollama",
        status=HealthStatus.HEALTHY,
        detail=f"model available: {settings.generation_model}",
    )


def run_health_checks(settings: Settings) -> tuple[HealthResult, ...]:
    """Run all external-service health checks."""

    with httpx.Client(timeout=settings.health_timeout_seconds) as client:
        return (
            check_qdrant(client, settings),
            check_ollama(client, settings),
        )


def main() -> int:
    try:
        settings = Settings()
    except ValidationError as exc:
        print(f"Configuration error: {exc}")
        return 2

    results = run_health_checks(settings)

    for result in results:
        print(f"{result.status.value.upper():<9} {result.service}: {result.detail}")

    return 0 if all(result.status is HealthStatus.HEALTHY for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())