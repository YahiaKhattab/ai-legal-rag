from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class CreateConversationRequest(BaseModel):
    """Request model for creating a new conversation."""

    title: str = Field(
        default="New conversation",
        min_length=1,
        max_length=200,
        description="Conversation title.",
    )


class RenameConversationRequest(BaseModel):
    """Request model for renaming an existing conversation."""

    title: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="New conversation title.",
    )


class ConversationResponse(BaseModel):
    """Conversation metadata returned by the API."""

    conversation_id: UUID
    title: str
    created_at: datetime
    updated_at: datetime


class ConversationMessage(BaseModel):
    """A single message returned as part of conversation history."""

    message_id: UUID
    role: str
    content: str
    timestamp: datetime


class ConversationHistoryResponse(BaseModel):
    """Full conversation history returned by the API."""

    conversation_id: UUID
    title: str
    created_at: datetime
    updated_at: datetime
    messages: list[ConversationMessage] = Field(
        default_factory=list
    )


class AskRequest(BaseModel):
    """Request model for legal question answering."""

    query: str = Field(
        ...,
        min_length=1,
        description="Natural-language legal question.",
    )

    conversation_id: UUID = Field(
        ...,
        description="Unique conversation identifier.",
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

    conversation_id: UUID

    question: str
    answer: str

    selected_legal_evidence: list[LegalEvidence] = Field(
        default_factory=list
    )

    citations: list[CitationResponse] = Field(
        default_factory=list
    )