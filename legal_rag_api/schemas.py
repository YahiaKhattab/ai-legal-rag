"""HTTP request and response models for the AI Legal RAG service."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class AskRequest(BaseModel):
    """Request model for legal question answering."""

    model_config = ConfigDict(populate_by_name=True)

    query: str = Field(
        ...,
        alias="legal-rag-query",
        min_length=1,
    )


class AskResponse(BaseModel):
    """Response model for legal question answering."""

    answer: str
    source: str