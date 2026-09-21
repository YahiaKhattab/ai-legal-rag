"""Fail-fast validation and stable identity for supported local documents."""

from __future__ import annotations

import hashlib
import zipfile
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from legal_rag.ingestion.models import SourceFormat

DEFAULT_MAXIMUM_PDF_BYTES = 100 * 1024 * 1024
DEFAULT_MAXIMUM_DOCUMENT_BYTES = DEFAULT_MAXIMUM_PDF_BYTES
_SOURCE_FORMAT_BY_SUFFIX = {
    ".pdf": SourceFormat.PDF,
    ".docx": SourceFormat.DOCX,
    ".txt": SourceFormat.TXT,
}


class PdfValidationError(ValueError):
    """Raised when an input is not a safe, readable PDF for this pipeline."""


class DocumentValidationError(ValueError):
    """Raised when a supported input is missing, unsafe, or unreadable."""


@dataclass(frozen=True, slots=True)
class ValidatedDocument:
    """Validated file facts reused by extraction, deduplication, and reporting."""

    path: Path
    file_size_bytes: int
    file_hash: str
    source_format: SourceFormat


ValidatedPdf = ValidatedDocument


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_pdf(
    path: Path,
    *,
    maximum_file_size_bytes: int = DEFAULT_MAXIMUM_PDF_BYTES,
) -> ValidatedDocument:
    """Validate type, size, signature, encryption, and basic readability."""

    resolved = path.resolve()
    if maximum_file_size_bytes <= 0:
        raise ValueError("maximum_file_size_bytes must be positive")
    if not resolved.is_file():
        raise PdfValidationError(f"PDF file does not exist: {resolved}")
    if resolved.suffix.lower() != ".pdf":
        raise PdfValidationError(f"File must use the .pdf extension: {resolved}")

    file_size = resolved.stat().st_size
    if file_size == 0:
        raise PdfValidationError(f"PDF file is empty: {resolved}")
    if file_size > maximum_file_size_bytes:
        raise PdfValidationError(
            f"PDF exceeds the configured size limit "
            f"({file_size} > {maximum_file_size_bytes} bytes): {resolved}"
        )

    try:
        with resolved.open("rb") as source:
            if source.read(5) != b"%PDF-":
                raise PdfValidationError(f"File does not have a PDF header: {resolved}")
            source.seek(0)
            reader = PdfReader(source, strict=False)
            if reader.is_encrypted:
                raise PdfValidationError(f"Encrypted PDFs are not supported: {resolved}")
            if len(reader.pages) == 0:
                raise PdfValidationError(f"PDF contains no readable pages: {resolved}")
    except PdfValidationError:
        raise
    except (OSError, PdfReadError, ValueError) as error:
        raise PdfValidationError(f"Unreadable PDF ({type(error).__name__}): {resolved}") from error

    return ValidatedDocument(
        path=resolved,
        file_size_bytes=file_size,
        file_hash=_sha256(resolved),
        source_format=SourceFormat.PDF,
    )


def _validate_common(path: Path, maximum_file_size_bytes: int) -> tuple[Path, int]:
    resolved = path.resolve()
    if maximum_file_size_bytes <= 0:
        raise ValueError("maximum_file_size_bytes must be positive")
    if not resolved.is_file():
        raise DocumentValidationError(f"Document file does not exist: {resolved}")
    if resolved.suffix.lower() not in _SOURCE_FORMAT_BY_SUFFIX:
        supported = ", ".join(sorted(_SOURCE_FORMAT_BY_SUFFIX))
        raise DocumentValidationError(
            f"Unsupported document extension {resolved.suffix!r}; expected: {supported}"
        )
    size = resolved.stat().st_size
    if size == 0:
        raise DocumentValidationError(f"Document file is empty: {resolved}")
    if size > maximum_file_size_bytes:
        raise DocumentValidationError(
            f"Document exceeds the configured size limit "
            f"({size} > {maximum_file_size_bytes} bytes): {resolved}"
        )
    return resolved, size


def _validate_docx(path: Path, maximum_expanded_bytes: int) -> None:
    if not zipfile.is_zipfile(path):
        raise DocumentValidationError(f"DOCX is not a valid Open XML package: {path}")
    try:
        with zipfile.ZipFile(path) as package:
            names = set(package.namelist())
            required = {"[Content_Types].xml", "word/document.xml"}
            if not required <= names:
                raise DocumentValidationError(
                    f"DOCX package is missing required document parts: {path}"
                )
            if any(info.flag_bits & 0x1 for info in package.infolist()):
                raise DocumentValidationError(f"Password-protected DOCX is not supported: {path}")
            expanded_size = sum(info.file_size for info in package.infolist())
            if expanded_size > maximum_expanded_bytes:
                raise DocumentValidationError(
                    f"DOCX exceeds the safe expanded-size limit "
                    f"({expanded_size} > {maximum_expanded_bytes} bytes): {path}"
                )
            package.read("word/document.xml")
    except DocumentValidationError:
        raise
    except (OSError, KeyError, RuntimeError, zipfile.BadZipFile) as error:
        raise DocumentValidationError(
            f"Unreadable DOCX ({type(error).__name__}): {path}"
        ) from error


def _validate_txt(path: Path) -> None:
    try:
        data = path.read_bytes()
        text = data.decode("utf-8-sig")
    except (OSError, UnicodeDecodeError) as error:
        raise DocumentValidationError(
            f"TXT must be readable UTF-8 ({type(error).__name__}): {path}"
        ) from error
    if "\x00" in text:
        raise DocumentValidationError(f"TXT contains binary NUL characters: {path}")
    if not text.strip():
        raise DocumentValidationError(f"TXT contains no non-whitespace text: {path}")


def validate_document(
    path: Path,
    *,
    maximum_file_size_bytes: int = DEFAULT_MAXIMUM_DOCUMENT_BYTES,
) -> ValidatedDocument:
    """Validate PDF, DOCX, or UTF-8 TXT input and compute its content hash."""

    resolved, size = _validate_common(path, maximum_file_size_bytes)
    source_format = _SOURCE_FORMAT_BY_SUFFIX[resolved.suffix.lower()]
    if source_format is SourceFormat.PDF:
        try:
            return validate_pdf(resolved, maximum_file_size_bytes=maximum_file_size_bytes)
        except PdfValidationError as error:
            raise DocumentValidationError(str(error)) from error
    if source_format is SourceFormat.DOCX:
        _validate_docx(resolved, maximum_file_size_bytes)
    else:
        _validate_txt(resolved)
    return ValidatedDocument(
        path=resolved,
        file_size_bytes=size,
        file_hash=_sha256(resolved),
        source_format=source_format,
    )
