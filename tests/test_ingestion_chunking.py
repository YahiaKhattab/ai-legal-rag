from legal_rag.ingestion.chunking import chunk_page
from legal_rag.ingestion.models import ExtractionMethod, PageRecord, TextQuality

_QUALITY = TextQuality(200, 0.8, 0.0, 0.0, 0.0)


def _page(text: str) -> PageRecord:
    return PageRecord(
        source_file="law.pdf",
        document_sha256="a" * 64,
        page_number=7,
        extraction_method=ExtractionMethod.NATIVE,
        raw_text=text,
        normalized_text=text,
        native_quality=_QUALITY,
        selected_quality=_QUALITY,
    )


def test_chunks_stay_on_one_page_and_have_stable_ids() -> None:
    page = _page(" ".join(f"word{index}" for index in range(100)))

    first = chunk_page(page, maximum_characters=120, overlap_characters=20)
    second = chunk_page(page, maximum_characters=120, overlap_characters=20)

    assert len(first) > 1
    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert all(chunk.page_number == 7 for chunk in first)
    assert all(len(chunk.text) <= 120 for chunk in first)


def test_failed_page_produces_no_chunks() -> None:
    page = _page("")
    failed = PageRecord(
        source_file=page.source_file,
        document_sha256=page.document_sha256,
        page_number=page.page_number,
        extraction_method=ExtractionMethod.FAILED,
        raw_text="",
        normalized_text="",
        native_quality=_QUALITY,
        selected_quality=_QUALITY,
        error="failure",
    )

    assert chunk_page(failed) == []
