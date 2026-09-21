from pathlib import Path

import pymupdf
import pytest
from docx import Document

from legal_rag.ingestion.models import SourceFormat
from legal_rag.ingestion.validation import (
    DocumentValidationError,
    PdfValidationError,
    validate_document,
    validate_pdf,
)


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


def test_accepts_docx_and_utf8_txt(tmp_path: Path) -> None:
    docx_path = tmp_path / "law.docx"
    document = Document()
    document.add_paragraph("Legal text")
    document.save(str(docx_path))
    txt_path = tmp_path / "law.txt"
    txt_path.write_text("Legal text", encoding="utf-8")

    assert validate_document(docx_path).source_format is SourceFormat.DOCX
    assert validate_document(txt_path).source_format is SourceFormat.TXT


@pytest.mark.parametrize(
    ("name", "content", "message"),
    [
        ("fake.docx", b"not a zip package", "Open XML package"),
        ("binary.txt", b"legal\x00text", "binary NUL"),
        ("legacy.txt", b"\xff\xfelegacy", "UTF-8"),
    ],
)
def test_rejects_malformed_non_pdf_documents(
    tmp_path: Path,
    name: str,
    content: bytes,
    message: str,
) -> None:
    path = tmp_path / name
    path.write_bytes(content)

    with pytest.raises(DocumentValidationError, match=message):
        validate_document(path)


def test_rejects_unsupported_extension(tmp_path: Path) -> None:
    path = tmp_path / "legacy.doc"
    path.write_bytes(b"legacy")

    with pytest.raises(DocumentValidationError, match="Unsupported document extension"):
        validate_document(path)
