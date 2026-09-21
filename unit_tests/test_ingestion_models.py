"""Unit tests for the dataclasses/enums in legal_rag.ingestion.models.

Most of these classes are plain data containers, but a few (ChunkingConfig,
SourceSegment) carry `__post_init__` validation that is genuine business
logic worth testing directly.
"""

from __future__ import annotations

import pytest

from legal_rag.ingestion.models import (
    ChunkingConfig,
    ExtractionMethod,
    SourceSegment,
)


def test_chunking_config_defaults_are_valid():
    config = ChunkingConfig()
    assert config.target_tokens == 400
    assert config.overlap_tokens == 60
    assert config.maximum_tokens == 480


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_tokens": 0},
        {"target_tokens": -10},
    ],
)
def test_chunking_config_rejects_non_positive_target(kwargs):
    with pytest.raises(ValueError, match="target_tokens must be positive"):
        ChunkingConfig(**kwargs)


def test_chunking_config_rejects_overlap_greater_or_equal_to_target():
    with pytest.raises(ValueError, match="overlap_tokens"):
        ChunkingConfig(target_tokens=100, overlap_tokens=100)


def test_chunking_config_rejects_negative_overlap():
    with pytest.raises(ValueError, match="overlap_tokens"):
        ChunkingConfig(target_tokens=100, overlap_tokens=-1)


def test_chunking_config_rejects_maximum_below_target_or_above_511():
    with pytest.raises(ValueError, match="maximum_tokens"):
        ChunkingConfig(target_tokens=400, overlap_tokens=60, maximum_tokens=300)
    with pytest.raises(ValueError, match="maximum_tokens"):
        ChunkingConfig(target_tokens=400, overlap_tokens=60, maximum_tokens=512)


def test_source_segment_valid_construction():
    segment = SourceSegment(
        start_char=0, end_char=10, locator_start=1, locator_end=1, kind="page"
    )
    assert segment.start_char == 0
    assert segment.end_char == 10


def test_source_segment_rejects_negative_start_char():
    with pytest.raises(ValueError, match="character range"):
        SourceSegment(start_char=-1, end_char=10, locator_start=1, locator_end=1, kind="page")


def test_source_segment_rejects_end_before_start():
    with pytest.raises(ValueError, match="character range"):
        SourceSegment(start_char=10, end_char=5, locator_start=1, locator_end=1, kind="page")


def test_source_segment_rejects_non_positive_locator_start():
    with pytest.raises(ValueError, match="locator range"):
        SourceSegment(start_char=0, end_char=10, locator_start=0, locator_end=1, kind="page")


def test_source_segment_rejects_locator_end_before_start():
    with pytest.raises(ValueError, match="locator range"):
        SourceSegment(start_char=0, end_char=10, locator_start=5, locator_end=1, kind="page")


def test_extraction_method_is_a_string_enum_with_expected_members():
    assert ExtractionMethod.NATIVE == "native"
    assert ExtractionMethod.OCR == "ocr"
    assert set(ExtractionMethod) == {
        ExtractionMethod.NATIVE,
        ExtractionMethod.OCR,
        ExtractionMethod.DOCX,
        ExtractionMethod.TXT,
        ExtractionMethod.FAILED,
    }
