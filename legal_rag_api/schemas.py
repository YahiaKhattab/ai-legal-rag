from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Request model for legal question answering."""

    query: str = Field(
        ...,
        min_length=1,
        description="Natural-language legal question.",
    )


class LegalEvidence(BaseModel):
    """Selected legal evidence shown to the API consumer."""

    citation: str
    source: str
    page: int | None = None
    section: str | None = None
    evidence: str


class CitationResponse(BaseModel):
    """Citation metadata shown to the API consumer."""

    marker: str
    source: str
    section: str | None = None
    page: int | None = None


class AskResponse(BaseModel):
    """User-facing response for legal question answering."""

    question: str
    answer: str

    selected_legal_evidence: list[LegalEvidence] = Field(
        default_factory=list
    )

    citations: list[CitationResponse] = Field(
        default_factory=list
    )