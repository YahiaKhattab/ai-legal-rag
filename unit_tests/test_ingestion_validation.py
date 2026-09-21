"""Unit tests for legal_rag.ingestion.validation.

These tests build tiny real PDF/DOCX/TXT files on disk with pypdf /
python-docx so validation runs against real file bytes instead of mocks
(this code is fundamentally about parsing file formats, so faking the
library calls would test very little).
"""

from __future__ import annotations

import zipfile

import pytest
from pypdf import PdfWriter

from legal_rag.ingestion.models import SourceFormat
from legal_rag.ingestion.validation import (
    DocumentValidationError,
    PdfValidationError,
    validate_document,
    validate_pdf,
)


def _make_pdf(path, *, pages=1, encrypted=False):
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=200, height=200)
    if encrypted:
        writer.encrypt("secret")
    with path.open("wb") as handle:
        writer.write(handle)


def _make_docx(path, *, valid=True):
    if valid:
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("[Content_Types].xml", "<Types/>")
            archive.writestr("word/document.xml", "<w:document/>")
    else:
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("[Content_Types].xml", "<Types/>")
            # Missing word/document.xml on purpose.


# --------------------------------------------------------------------- PDF


def test_validate_pdf_accepts_a_real_pdf(tmp_path):
    pdf_path = tmp_path / "doc.pdf"
    _make_pdf(pdf_path)

    result = validate_pdf(pdf_path)

    assert result.source_format is SourceFormat.PDF
    assert result.file_size_bytes > 0
    assert len(result.file_hash) == 64  # sha256 hex digest length


def test_validate_pdf_rejects_missing_file(tmp_path):
    with pytest.raises(PdfValidationError, match="does not exist"):
        validate_pdf(tmp_path / "missing.pdf")


def test_validate_pdf_rejects_wrong_extension(tmp_path):
    path = tmp_path / "doc.txt"
    path.write_bytes(b"%PDF-1.4 fake")
    with pytest.raises(PdfValidationError, match="\\.pdf extension"):
        validate_pdf(path)


def test_validate_pdf_rejects_empty_file(tmp_path):
    path = tmp_path / "empty.pdf"
    path.write_bytes(b"")
    with pytest.raises(PdfValidationError, match="empty"):
        validate_pdf(path)


def test_validate_pdf_rejects_file_over_size_limit(tmp_path):
    path = tmp_path / "doc.pdf"
    _make_pdf(path)
    with pytest.raises(PdfValidationError, match="exceeds the configured size limit"):
        validate_pdf(path, maximum_file_size_bytes=1)


def test_validate_pdf_rejects_bad_header(tmp_path):
    path = tmp_path / "doc.pdf"
    path.write_bytes(b"NOT A PDF HEADER AT ALL")
    with pytest.raises(PdfValidationError, match="PDF header"):
        validate_pdf(path)


def test_validate_pdf_rejects_encrypted_pdf(tmp_path):
    path = tmp_path / "doc.pdf"
    _make_pdf(path, encrypted=True)
    with pytest.raises(PdfValidationError, match="Encrypted"):
        validate_pdf(path)


def test_validate_pdf_rejects_zero_page_document(tmp_path):
    path = tmp_path / "doc.pdf"
    writer = PdfWriter()
    with path.open("wb") as handle:
        writer.write(handle)
    with pytest.raises(PdfValidationError, match="no readable pages"):
        validate_pdf(path)


def test_validate_pdf_rejects_non_positive_size_limit(tmp_path):
    path = tmp_path / "doc.pdf"
    _make_pdf(path)
    with pytest.raises(ValueError, match="must be positive"):
        validate_pdf(path, maximum_file_size_bytes=0)


# ---------------------------------------------------------------- DOCX/TXT


def test_validate_document_accepts_valid_docx(tmp_path):
    path = tmp_path / "doc.docx"
    _make_docx(path, valid=True)

    result = validate_document(path)

    assert result.source_format is SourceFormat.DOCX


def test_validate_document_rejects_docx_missing_required_parts(tmp_path):
    path = tmp_path / "doc.docx"
    _make_docx(path, valid=False)

    with pytest.raises(DocumentValidationError, match="missing required document parts"):
        validate_document(path)


def test_validate_document_rejects_docx_that_is_not_a_zip(tmp_path):
    path = tmp_path / "doc.docx"
    path.write_bytes(b"not a zip file")

    with pytest.raises(DocumentValidationError, match="not a valid Open XML package"):
        validate_document(path)


def test_validate_document_accepts_valid_txt(tmp_path):
    path = tmp_path / "doc.txt"
    path.write_text("Article 1: content", encoding="utf-8")

    result = validate_document(path)

    assert result.source_format is SourceFormat.TXT


def test_validate_document_rejects_empty_txt(tmp_path):
    path = tmp_path / "doc.txt"
    path.write_text("   \n  ", encoding="utf-8")

    with pytest.raises(DocumentValidationError, match="no non-whitespace text"):
        validate_document(path)


def test_validate_document_rejects_txt_with_nul_bytes(tmp_path):
    path = tmp_path / "doc.txt"
    path.write_bytes("hello\x00world".encode("utf-8"))

    with pytest.raises(DocumentValidationError, match="binary NUL"):
        validate_document(path)


def test_validate_document_rejects_non_utf8_txt(tmp_path):
    path = tmp_path / "doc.txt"
    path.write_bytes("مرحبا".encode("windows-1256"))

    with pytest.raises(DocumentValidationError, match="UTF-8"):
        validate_document(path)


def test_validate_document_rejects_unsupported_extension(tmp_path):
    path = tmp_path / "doc.rtf"
    path.write_text("hi", encoding="utf-8")

    with pytest.raises(DocumentValidationError, match="Unsupported document extension"):
        validate_document(path)


def test_validate_document_rejects_file_over_size_limit(tmp_path):
    path = tmp_path / "doc.txt"
    path.write_text("hello world", encoding="utf-8")

    with pytest.raises(DocumentValidationError, match="exceeds the configured size limit"):
        validate_document(path, maximum_file_size_bytes=1)


def test_validate_document_delegates_pdf_errors_with_document_error_type(tmp_path):
    path = tmp_path / "doc.pdf"
    path.write_bytes(b"not really a pdf")

    with pytest.raises(DocumentValidationError):
        validate_document(path)
