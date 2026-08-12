from pathlib import Path

import pymupdf
import pytest

from legal_rag.ingestion.validation import PdfValidationError, validate_pdf


def _make_pdf(path: Path) -> None:
    with pymupdf.open() as document:  # type: ignore[no-untyped-call]
        document.new_page()
        document.save(path)


def test_accepts_readable_pdf_and_computes_hash(tmp_path: Path) -> None:
    path = tmp_path / "law.pdf"
    _make_pdf(path)

    result = validate_pdf(path)

    assert result.path == path.resolve()
    assert result.file_size_bytes == path.stat().st_size
    assert len(result.file_hash) == 64


def test_rejects_renamed_non_pdf_content(tmp_path: Path) -> None:
    path = tmp_path / "fake.pdf"
    path.write_text("not a PDF", encoding="utf-8")

    with pytest.raises(PdfValidationError, match="PDF header"):
        validate_pdf(path)


def test_rejects_pdf_over_size_limit(tmp_path: Path) -> None:
    path = tmp_path / "law.pdf"
    _make_pdf(path)

    with pytest.raises(PdfValidationError, match="size limit"):
        validate_pdf(path, maximum_file_size_bytes=path.stat().st_size - 1)
