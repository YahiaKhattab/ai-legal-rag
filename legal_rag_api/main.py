"""FastAPI application entry point for the AI Legal RAG service."""

from __future__ import annotations

from fastapi import FastAPI

from legal_rag_api.routers.legal_ai import router as legal_ai_router


app = FastAPI(
    title="AI Legal RAG Service",
    version="0.1.0",
)

app.include_router(legal_ai_router)


@app.get("/health")
async def health() -> dict[str, str]:
    """Return the health status of the API service."""

    return {"status": "ok"}