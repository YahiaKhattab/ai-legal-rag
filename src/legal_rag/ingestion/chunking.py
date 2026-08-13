"""Structure-aware, token-bounded chunking for precise legal citations."""

from __future__ import annotations

import hashlib

from legal_rag.ingestion.models import (
    ChunkingConfig,
    ChunkRecord,
    DocumentMetadata,
    SourceRecord,
)
from legal_rag.ingestion.normalization import normalize_text
from legal_rag.ingestion.quality import detect_language, measure_text_quality
from legal_rag.ingestion.structure import SectionSpan
from legal_rag.ingestion.tokenization import TokenCounter

PIPELINE_VERSION = "1.2.0"
DEFAULT_CHUNKING_CONFIG = ChunkingConfig()
_SENTENCE_ENDINGS = frozenset("\n.؟!؛;")


def _normalized_slice(source: SourceRecord, start: int, end: int) -> str:
    return normalize_text(
        source.original_text[start:end],
        reverse_arabic_digit_runs=source.native_rtl_digit_correction_applied,
    )


def _maximum_end(
    page: SourceRecord,
    *,
    start: int,
    section_end: int,
    maximum_tokens: int,
    token_counter: TokenCounter,
) -> int:
    """Find the greatest character end that fits the hard passage limit."""

    low = start + 1
    high = section_end
    best = start
    while low <= high:
        middle = (low + high) // 2
        normalized = _normalized_slice(page, start, middle)
        if token_counter.count_passage(normalized) <= maximum_tokens:
            best = middle
            low = middle + 1
        else:
            high = middle - 1
    if best == start:
        raise ValueError("One source character exceeds the configured token maximum")
    return best


def _candidate_breaks(
    text: str,
    *,
    start: int,
    hard_end: int,
    preferred_breaks: tuple[int, ...],
) -> list[int]:
    structural = [position for position in preferred_breaks if start < position <= hard_end]
    sentences = [index + 1 for index in range(start, hard_end) if text[index] in _SENTENCE_ENDINGS]
    whitespace = [index + 1 for index in range(start, hard_end) if text[index].isspace()]
    return sorted(set([*structural, *sentences, *whitespace, hard_end]))


def _choose_end(
    page: SourceRecord,
    span: SectionSpan,
    *,
    start: int,
    config: ChunkingConfig,
    token_counter: TokenCounter,
) -> int:
    hard_end = _maximum_end(
        page,
        start=start,
        section_end=span.end_char,
        maximum_tokens=config.maximum_tokens,
        token_counter=token_counter,
    )
    if hard_end == span.end_char:
        return hard_end

    candidates = _candidate_breaks(
        page.original_text,
        start=start,
        hard_end=hard_end,
        preferred_breaks=span.preferred_breaks,
    )
    target_candidates = [
        end
        for end in candidates
        if token_counter.count_passage(_normalized_slice(page, start, end)) <= config.target_tokens
    ]
    if target_candidates:
        return target_candidates[-1]
    return hard_end


def _next_start(
    page: SourceRecord,
    *,
    section_start: int,
    end: int,
    overlap_tokens: int,
    token_counter: TokenCounter,
) -> int:
    if overlap_tokens == 0:
        return end

    low = section_start
    high = end
    best = end
    while low <= high:
        middle = (low + high) // 2
        overlap = _normalized_slice(page, middle, end)
        if token_counter.count_content(overlap) <= overlap_tokens:
            best = middle
            high = middle - 1
        else:
            low = middle + 1

    while best < end and best > section_start and not page.original_text[best - 1].isspace():
        best += 1
    return min(best, end)


def _locator_range(source: SourceRecord, start: int, end: int) -> tuple[int, int]:
    intersecting = [
        segment
        for segment in source.source_segments
        if segment.end_char > start and segment.start_char < end
    ]
    if not intersecting:
        return source.locator_start, source.locator_end
    return intersecting[0].locator_start, intersecting[-1].locator_end


def chunk_source_sections(
    page: SourceRecord,
    spans: list[SectionSpan],
    metadata: DocumentMetadata,
    token_counter: TokenCounter,
    *,
    starting_index: int = 0,
    config: ChunkingConfig = DEFAULT_CHUNKING_CONFIG,
) -> list[ChunkRecord]:
    """Split one source record without merging unrelated legal boundaries."""

    chunks: list[ChunkRecord] = []
    for span in spans:
        start = span.start_char
        while start < span.end_char:
            end = _choose_end(
                page,
                span,
                start=start,
                config=config,
                token_counter=token_counter,
            )
            original = page.original_text[start:end]
            left_trim = len(original) - len(original.lstrip())
            right_trim = len(original) - len(original.rstrip())
            actual_start = start + left_trim
            actual_end = end - right_trim
            original = page.original_text[actual_start:actual_end]
            normalized = _normalized_slice(page, actual_start, actual_end)

            if original and normalized:
                chunk_index = starting_index + len(chunks)
                identity = (
                    f"{metadata.document_id}:{metadata.document_version}:"
                    f"{PIPELINE_VERSION}:{page.source_format.value}:"
                    f"{page.locator_type.value}:{actual_start}:{actual_end}:"
                    f"{span.section_type.value}:{span.section_title or ''}"
                )
                locator_start, locator_end = _locator_range(page, actual_start, actual_end)
                token_count = token_counter.count_passage(normalized)
                if token_count > config.maximum_tokens:
                    raise AssertionError("Chunk exceeded the hard token maximum")
                language = page.language
                if language in ("mixed", "unknown"):
                    language = detect_language(measure_text_quality(normalized))
                chunks.append(
                    ChunkRecord(
                        chunk_id=hashlib.sha256(identity.encode("utf-8")).hexdigest(),
                        document_id=metadata.document_id,
                        document_version=metadata.document_version,
                        chunk_index=chunk_index,
                        original_text=original,
                        normalized_text=normalized,
                        section_type=span.section_type,
                        section_title=span.section_title,
                        page_start=page.page_number,
                        page_end=page.page_number,
                        source_format=page.source_format,
                        locator_type=page.locator_type,
                        locator_start=locator_start,
                        locator_end=locator_end,
                        language=language,
                        document_type=metadata.document_type,
                        source=metadata.source,
                        source_file=metadata.source_file,
                        file_hash=metadata.file_hash,
                        extraction_methods=(page.extraction_method,),
                        original_start_char=actual_start,
                        original_end_char=actual_end,
                        token_count=token_count,
                        tokenizer_name=token_counter.name,
                        pipeline_version=PIPELINE_VERSION,
                    )
                )

            if end >= span.end_char:
                break
            next_start = _next_start(
                page,
                section_start=span.start_char,
                end=end,
                overlap_tokens=config.overlap_tokens,
                token_counter=token_counter,
            )
            start = max(next_start, start + 1)

    return chunks


# Compatibility name retained for callers of the PDF-only pipeline.
chunk_page_sections = chunk_source_sections
