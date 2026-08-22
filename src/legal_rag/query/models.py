"""Typed data contracts for the retrieval and answer stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    document_id: str
    score: float
    text: str
    source_file: str | None = None
    section_type: str | None = None
    section_title: str | None = None
    page: int | None = None
    language: str | None = None
    document_type: str | None = None
    source: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RerankedChunk(RetrievedChunk):
    rerank_score: float = 0.0


@dataclass(frozen=True)
class Citation:
    marker: str
    chunk_id: str
    document_id: str
    source_file: str | None
    section_title: str | None
    page: int | None


@dataclass(frozen=True)
class LegalExcerpt:
    """Original legal text selected from a retrieved source chunk."""

    marker: str
    text: str
    source_file: str | None
    section_title: str | None
    page: int | None
    chunk_id: str


@dataclass(frozen=True)
class RetrievalDiagnostics:
    """Observable retrieval signals used by the sufficiency gate."""

    strategy: str
    candidate_count: int
    used_chunk_count: int
    sufficient: bool
    reason: str
    top_dense_score: float | None
    dense_score_margin: float | None
    top_rerank_score: float | None
    exact_identifier_match: bool
    source_count: int


@dataclass(frozen=True)
class CitedAnswer:
    query: str
    answer_text: str
    language: str
    citations: list[Citation]
    retrieved_chunk_ids: list[str]
    legal_excerpts: list[LegalExcerpt] = field(default_factory=list)
    retrieval: RetrievalDiagnostics | None = None
    prompt_version: str | None = None
