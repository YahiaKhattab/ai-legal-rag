"""Typed data contracts for the retrieval stage.

These models describe data flowing between Qdrant search, reranking,
prompting, and generation. Field names intentionally mirror the ChunkRecord /
Qdrant payload fields already documented for the ingestion and
vector-indexing stages, so nothing gets renamed on its way through the
pipeline.
"""
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
class CitedAnswer:
    query: str
    answer_text: str
    language: str
    citations: list[Citation]
    retrieved_chunk_ids: list[str]
