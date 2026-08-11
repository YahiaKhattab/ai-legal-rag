"""Data contracts for the ingestion pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class ExtractionMethod(StrEnum):
    """How the selected page text was obtained."""

    NATIVE = "native"
    OCR = "ocr"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class TextQuality:
    character_count: int
    arabic_ratio: float
    latin_ratio: float
    replacement_ratio: float
    control_ratio: float


@dataclass(frozen=True, slots=True)
class OcrText:
    text: str
    mean_confidence: float | None


@dataclass(frozen=True, slots=True)
class PageRecord:
    source_file: str
    document_sha256: str
    page_number: int
    extraction_method: ExtractionMethod
    raw_text: str
    normalized_text: str
    native_quality: TextQuality
    selected_quality: TextQuality
    ocr_mean_confidence: float | None = None
    error: str | None = None
    native_rtl_digit_correction_applied: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["extraction_method"] = self.extraction_method.value
        return data


@dataclass(frozen=True, slots=True)
class ChunkRecord:
    chunk_id: str
    source_file: str
    document_sha256: str
    page_number: int
    chunk_index: int
    extraction_method: ExtractionMethod
    start_char: int
    end_char: int
    text: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["extraction_method"] = self.extraction_method.value
        return data
