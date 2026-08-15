"""Read persisted chunk records from JSONL files."""

from __future__ import annotations

import json
from pathlib import Path

from legal_rag.ingestion.models import (
    ChunkRecord,
    ExtractionMethod,
    LocatorType,
    SectionType,
    SourceFormat,
)


def read_chunks(path: Path) -> list[ChunkRecord]:
    """Read chunk records from a UTF-8 JSONL file."""
    chunks: list[ChunkRecord] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} in {path}") from exc

            if not isinstance(data, dict):
                raise ValueError(f"Expected a JSON object on line {line_number} in {path}")

            chunks.append(
                ChunkRecord(
                    chunk_id=data["chunk_id"],
                    document_id=data["document_id"],
                    document_version=data["document_version"],
                    chunk_index=data["chunk_index"],
                    original_text=data["original_text"],
                    normalized_text=data["normalized_text"],
                    section_type=SectionType(data["section_type"]),
                    section_title=data["section_title"],
                    page_start=data["page_start"],
                    page_end=data["page_end"],
                    source_format=SourceFormat(data["source_format"]),
                    locator_type=LocatorType(data["locator_type"]),
                    locator_start=data["locator_start"],
                    locator_end=data["locator_end"],
                    language=data["language"],
                    document_type=data["document_type"],
                    source=data["source"],
                    source_file=data["source_file"],
                    file_hash=data["file_hash"],
                    extraction_methods=tuple(
                        ExtractionMethod(method) for method in data["extraction_methods"]
                    ),
                    original_start_char=data["original_start_char"],
                    original_end_char=data["original_end_char"],
                    token_count=data["token_count"],
                    tokenizer_name=data["tokenizer_name"],
                    pipeline_version=data["pipeline_version"],
                )
            )

    return chunks
