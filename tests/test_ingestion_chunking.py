# ruff: noqa: RUF001 - Arabic fixtures intentionally exercise real legal text

from legal_rag.ingestion.chunking import chunk_page_sections
from legal_rag.ingestion.models import (
    ChunkingConfig,
    DocumentMetadata,
    ExtractionMethod,
    PageRecord,
    SectionType,
    TextQuality,
)
from legal_rag.ingestion.structure import LegalStructureDetector

_QUALITY = TextQuality(200, 0.8, 0.0, 0.0, 0.0)


class WordTokenCounter:
    """Deterministic test double; production uses the exact E5 tokenizer."""

    name = "test-word-counter"

    def count_passage(self, text: str) -> int:
        return len(text.split()) + 3

    def count_content(self, text: str) -> int:
        return len(text.split())


def _page(text: str) -> PageRecord:
    return PageRecord(
        source_file="law.pdf",
        document_id="a" * 64,
        document_version=1,
        file_hash="a" * 64,
        page_number=7,
        extraction_method=ExtractionMethod.NATIVE,
        native_text=text,
        original_text=text,
        normalized_text=text,
        native_quality=_QUALITY,
        selected_quality=_QUALITY,
        language="ar",
    )


def _metadata() -> DocumentMetadata:
    return DocumentMetadata(
        document_id="a" * 64,
        document_version=1,
        document_type="law",
        source="Official Gazette",
        source_file="law.pdf",
        file_hash="a" * 64,
    )


def test_long_article_is_token_bounded_and_has_stable_ids() -> None:
    text = "المادة (١)\n" + " ".join(f"كلمة{index}" for index in range(140))
    page = _page(text)
    spans = LegalStructureDetector().detect_page(page)
    config = ChunkingConfig(target_tokens=30, overlap_tokens=5, maximum_tokens=36)

    first = chunk_page_sections(
        page,
        spans,
        _metadata(),
        WordTokenCounter(),
        config=config,
    )
    second = chunk_page_sections(
        page,
        spans,
        _metadata(),
        WordTokenCounter(),
        config=config,
    )

    assert len(first) > 1
    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert [chunk.chunk_index for chunk in first] == list(range(len(first)))
    assert all(chunk.token_count <= config.maximum_tokens for chunk in first)
    assert all(chunk.section_title == "المادة (١)" for chunk in first)
    assert all(chunk.page_start == chunk.page_end == 7 for chunk in first)


def test_unrelated_articles_are_never_merged() -> None:
    text = "المادة (١)\nنص المادة الأولى قصير.\nالمادة (٢)\nنص المادة الثانية مستقل."
    page = _page(text)
    spans = LegalStructureDetector().detect_page(page)

    chunks = chunk_page_sections(
        page,
        spans,
        _metadata(),
        WordTokenCounter(),
        config=ChunkingConfig(target_tokens=100, overlap_tokens=10, maximum_tokens=120),
    )

    assert [chunk.section_title for chunk in chunks] == ["المادة (١)", "المادة (٢)"]
    assert "المادة (٢)" not in chunks[0].original_text
    assert "المادة (١)" not in chunks[1].original_text


def test_chunk_original_text_matches_immutable_source_span() -> None:
    text = "المادة (١)\nإِنَّ النَّصَّ الأصلي يبقى كما هو."
    page = _page(text)
    spans = LegalStructureDetector().detect_page(page)

    [chunk] = chunk_page_sections(
        page,
        spans,
        _metadata(),
        WordTokenCounter(),
    )

    assert chunk.original_text == text
    assert page.original_text[chunk.original_start_char : chunk.original_end_char] == text
    assert chunk.normalized_text != chunk.original_text


def test_failed_page_produces_no_structure_or_chunks() -> None:
    page = _page("")
    failed = PageRecord(
        source_file=page.source_file,
        document_id=page.document_id,
        document_version=page.document_version,
        file_hash=page.file_hash,
        page_number=page.page_number,
        extraction_method=ExtractionMethod.FAILED,
        native_text="corrupted layer",
        original_text="",
        normalized_text="",
        native_quality=_QUALITY,
        selected_quality=_QUALITY,
        language="ar",
        error="failure",
    )

    assert LegalStructureDetector().detect_page(failed) == []
    assert SectionType.ARTICLE.value == "article"
