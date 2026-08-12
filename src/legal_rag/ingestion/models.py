"""Stable data contracts shared by ingestion pipeline stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class ExtractionMethod(StrEnum):
    """How the selected page text was obtained."""

    NATIVE = "native"
    OCR = "ocr"
    FAILED = "failed"


class IngestionStatus(StrEnum):
    """Whether a document was processed or reused by content identity."""

    PROCESSED = "processed"
    DUPLICATE = "duplicate"


class SectionType(StrEnum):
    """Legal section types recognized by the conservative structure detector."""

    DOCUMENT = "document"
    PART = "part"
    CHAPTER = "chapter"
    SECTION = "section"
    ARTICLE = "article"


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """Token limits applied to the exact E5 passage input."""

    target_tokens: int = 400
    overlap_tokens: int = 60
    maximum_tokens: int = 480

    def __post_init__(self) -> None:
        if self.target_tokens <= 0:
            raise ValueError("target_tokens must be positive")
        if not 0 <= self.overlap_tokens < self.target_tokens:
            raise ValueError("overlap_tokens must be between 0 and target_tokens")
        if not self.target_tokens <= self.maximum_tokens < 512:
            raise ValueError("maximum_tokens must be between target_tokens and 511")


@dataclass(frozen=True, slots=True)
class DocumentMetadata:
    """Document-level values copied into every persisted chunk."""

    document_id: str
    document_version: int
    document_type: str
    source: str
    source_file: str
    file_hash: str


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
    """Extraction evidence for one source page.

    native_text always preserves the PDF text layer. original_text is the exact
    text selected for downstream use: native text or OCR output. Normalization
    never mutates either field.
    """

    source_file: str
    document_id: str
    document_version: int
    file_hash: str
    page_number: int
    extraction_method: ExtractionMethod
    native_text: str
    original_text: str
    normalized_text: str
    native_quality: TextQuality
    selected_quality: TextQuality
    language: str
    ocr_mean_confidence: float | None = None
    error: str | None = None
    native_rtl_digit_correction_applied: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["extraction_method"] = self.extraction_method.value
        return data


@dataclass(frozen=True, slots=True)
class ChunkRecord:
    """Embedding-ready legal text plus citation and provenance metadata."""

    chunk_id: str
    document_id: str
    document_version: int
    chunk_index: int
    original_text: str
    normalized_text: str
    section_type: SectionType
    section_title: str | None
    page_start: int
    page_end: int
    language: str
    document_type: str
    source: str
    source_file: str
    file_hash: str
    extraction_methods: tuple[ExtractionMethod, ...]
    original_start_char: int
    original_end_char: int
    token_count: int
    tokenizer_name: str
    pipeline_version: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["section_type"] = self.section_type.value
        data["extraction_methods"] = [method.value for method in self.extraction_methods]
        return data
