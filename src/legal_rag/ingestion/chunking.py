"""Page-bounded chunking for precise source citations."""

from __future__ import annotations

import hashlib

from legal_rag.ingestion.models import ChunkRecord, ExtractionMethod, PageRecord

_BREAK_CHARACTERS = ("\n", "؟", ".", "!", "؛", " ")


def _choose_end(text: str, start: int, maximum_characters: int) -> int:
    hard_end = min(start + maximum_characters, len(text))
    if hard_end == len(text):
        return hard_end

    lower_bound = start + maximum_characters // 2
    candidates = [text.rfind(mark, lower_bound, hard_end) for mark in _BREAK_CHARACTERS]
    best = max(candidates)
    return best + 1 if best >= lower_bound else hard_end


def _next_start(text: str, end: int, overlap_characters: int) -> int:
    candidate = max(0, end - overlap_characters)
    if candidate == 0 or candidate >= len(text) or text[candidate - 1].isspace():
        return candidate
    next_space = text.find(" ", candidate, end)
    return next_space + 1 if next_space >= 0 else candidate


def chunk_page(
    page: PageRecord,
    *,
    maximum_characters: int = 1_200,
    overlap_characters: int = 150,
) -> list[ChunkRecord]:
    if maximum_characters <= 0:
        raise ValueError("maximum_characters must be positive")
    if not 0 <= overlap_characters < maximum_characters:
        raise ValueError("overlap_characters must be between 0 and maximum_characters")
    if page.extraction_method is ExtractionMethod.FAILED or not page.normalized_text:
        return []

    text = page.normalized_text
    chunks: list[ChunkRecord] = []
    start = 0

    while start < len(text):
        end = _choose_end(text, start, maximum_characters)
        left_trimmed = len(text[start:end]) - len(text[start:end].lstrip())
        right_trimmed = len(text[start:end]) - len(text[start:end].rstrip())
        actual_start = start + left_trimmed
        actual_end = end - right_trimmed
        chunk_text = text[actual_start:actual_end]

        if chunk_text:
            chunk_index = len(chunks)
            identity = (
                f"{page.document_sha256}:{page.page_number}:"
                f"{chunk_index}:{actual_start}:{actual_end}"
            )
            chunks.append(
                ChunkRecord(
                    chunk_id=hashlib.sha256(identity.encode("utf-8")).hexdigest(),
                    source_file=page.source_file,
                    document_sha256=page.document_sha256,
                    page_number=page.page_number,
                    chunk_index=chunk_index,
                    extraction_method=page.extraction_method,
                    start_char=actual_start,
                    end_char=actual_end,
                    text=chunk_text,
                )
            )

        if end >= len(text):
            break
        next_start = _next_start(text, end, overlap_characters)
        start = max(next_start, start + 1)

    return chunks
