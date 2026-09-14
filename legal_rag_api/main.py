"""FastAPI application entry point for the AI Legal RAG service."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from legal_rag.observability.tracing import initialize, shutdown
from legal_rag_api.routers.legal_ai import router as legal_ai_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    initialize()
    try:
        yield
    finally:
        shutdown()


app = FastAPI(
    lifespan=lifespan,
    title="AI Legal RAG Service",
    version="0.1.0",
)

app.include_router(legal_ai_router)


@app.get("/health")
async def health() -> dict[str, str]:
    """Return the health status of the API service."""

    return {"status": "ok"}