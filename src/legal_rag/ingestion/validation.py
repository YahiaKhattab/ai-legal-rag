"""Fail-fast validation for local PDF inputs."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

DEFAULT_MAXIMUM_PDF_BYTES = 100 * 1024 * 1024


class PdfValidationError(ValueError):
    """Raised when an input is not a safe, readable PDF for this pipeline."""


@dataclass(frozen=True, slots=True)
class ValidatedPdf:
    """Validated file facts reused by deduplication and reporting."""

    path: Path
    file_size_bytes: int
    file_hash: str


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
) -> ValidatedPdf:
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

    return ValidatedPdf(
        path=resolved,
        file_size_bytes=file_size,
        file_hash=_sha256(resolved),
    )
