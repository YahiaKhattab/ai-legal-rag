"""Unit tests for legal_rag.ingestion.chunking.chunk_source_sections.

Chunking is the most algorithmically complex piece of the ingestion
pipeline (binary search over token boundaries + preferred break points),
so instead of using the real E5 tokenizer we use a tiny, fully
deterministic fake TokenCounter: "1 token per character". That makes the
expected chunk boundaries easy to reason about and predict by hand.
"""

from __future__ import annotations

from legal_rag.ingestion.chunking import chunk_source_sections
from legal_rag.ingestion.models import (
    ChunkingConfig,
    DocumentMetadata,
    ExtractionMethod,
    LocatorType,
    SectionType,
    SourceFormat,
    SourceRecord,
    TextQuality,
)
from legal_rag.ingestion.structure import SectionSpan

_QUALITY = TextQuality(
    character_count=1, arabic_ratio=0.0, latin_ratio=1.0, replacement_ratio=0.0, control_ratio=0.0
)


class _OneTokenPerCharCounter:
    """Fake TokenCounter: token count == character count (prefix ignored)."""

    name = "fake-1-char-1-token"

    def count_passage(self, text: str) -> int:
        return len(text)

    def count_content(self, text: str) -> int:
        return len(text)


def _page(text: str) -> SourceRecord:
    return SourceRecord(
        source_file="doc.pdf",
        document_id="doc-1",
        document_version=1,
        file_hash="hash",
        extraction_method=ExtractionMethod.NATIVE,
        native_text=text,
        original_text=text,
        normalized_text=text,
        native_quality=_QUALITY,
        selected_quality=_QUALITY,
        language="en",
        source_format=SourceFormat.PDF,
        locator_type=LocatorType.PAGE,
        locator_start=1,
        locator_end=1,
        page_number=1,
    )


def _metadata() -> DocumentMetadata:
    return DocumentMetadata(
        document_id="doc-1",
        document_version=1,
        document_type="statute",
        source="test-suite",
        source_file="doc.pdf",
        file_hash="hash",
        source_format=SourceFormat.PDF,
    )


def _whole_document_span(text: str) -> SectionSpan:
    return SectionSpan(
        page_number=1,
        start_char=0,
        end_char=len(text),
        section_type=SectionType.DOCUMENT,
        section_title=None,
        preferred_breaks=(),
    )


def test_short_text_produces_a_single_chunk():
    text = "This is a short legal sentence."
    page = _page(text)
    config = ChunkingConfig(target_tokens=400, overlap_tokens=60, maximum_tokens=480)

    chunks = chunk_source_sections(
        page, [_whole_document_span(text)], _metadata(), _OneTokenPerCharCounter(), config=config
    )

    assert len(chunks) == 1
    assert chunks[0].original_text == text
    assert chunks[0].chunk_index == 0


def test_long_text_is_split_into_multiple_chunks():
    # 1 token == 1 char here, so target_tokens=20 forces multiple chunks
    # for a ~100 character sentence-separated text.
    sentence = "Word word word word word. "  # 27 chars, ends with a sentence break
    text = sentence * 10  # ~270 characters
    page = _page(text)
    config = ChunkingConfig(target_tokens=20, overlap_tokens=5, maximum_tokens=30)

    chunks = chunk_source_sections(
        page, [_whole_document_span(text)], _metadata(), _OneTokenPerCharCounter(), config=config
    )

    assert len(chunks) > 1
    # Chunk indices must be sequential starting at 0.
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    # No chunk may exceed the hard token maximum.
    assert all(chunk.token_count <= config.maximum_tokens for chunk in chunks)


def test_starting_index_offsets_chunk_indices():
    text = "Some short text."
    page = _page(text)

    chunks = chunk_source_sections(
        page,
        [_whole_document_span(text)],
        _metadata(),
        _OneTokenPerCharCounter(),
        starting_index=5,
    )

    assert chunks[0].chunk_index == 5


def test_chunk_carries_metadata_and_provenance_fields():
    text = "Some legal text."
    page = _page(text)
    metadata = _metadata()

    chunks = chunk_source_sections(
        page, [_whole_document_span(text)], metadata, _OneTokenPerCharCounter()
    )

    chunk = chunks[0]
    assert chunk.document_id == metadata.document_id
    assert chunk.document_type == metadata.document_type
    assert chunk.source == metadata.source
    assert chunk.source_file == metadata.source_file
    assert chunk.file_hash == metadata.file_hash
    assert chunk.extraction_methods == (ExtractionMethod.NATIVE,)
    assert chunk.tokenizer_name == "fake-1-char-1-token"


def test_empty_spans_produce_no_chunks():
    text = ""
    page = _page(text)
    chunks = chunk_source_sections(page, [], _metadata(), _OneTokenPerCharCounter())
    assert chunks == []


def test_whitespace_only_span_produces_no_chunk():
    text = "     "
    page = _page(text)
    span = _whole_document_span(text)
    chunks = chunk_source_sections(page, [span], _metadata(), _OneTokenPerCharCounter())
    assert chunks == []


def test_chunk_ids_are_stable_for_identical_input():
    text = "Repeatable legal content."
    page = _page(text)
    span = _whole_document_span(text)

    chunks_a = chunk_source_sections(page, [span], _metadata(), _OneTokenPerCharCounter())
    chunks_b = chunk_source_sections(page, [span], _metadata(), _OneTokenPerCharCounter())

    assert chunks_a[0].chunk_id == chunks_b[0].chunk_id


def test_different_document_version_changes_chunk_id():
    text = "Some content that stays the same."
    page = _page(text)
    span = _whole_document_span(text)
    metadata_v1 = _metadata()
    metadata_v2 = DocumentMetadata(
        document_id=metadata_v1.document_id,
        document_version=2,
        document_type=metadata_v1.document_type,
        source=metadata_v1.source,
        source_file=metadata_v1.source_file,
        file_hash=metadata_v1.file_hash,
        source_format=metadata_v1.source_format,
    )

    chunk_v1 = chunk_source_sections(page, [span], metadata_v1, _OneTokenPerCharCounter())[0]
    chunk_v2 = chunk_source_sections(page, [span], metadata_v2, _OneTokenPerCharCounter())[0]

    assert chunk_v1.chunk_id != chunk_v2.chunk_id


def test_chunk_page_sections_alias_matches_chunk_source_sections():
    from legal_rag.ingestion.chunking import chunk_page_sections

    assert chunk_page_sections is chunk_source_sections
