"""Unit tests for legal_rag.ingestion.ocr.PaddleOcrEngine.

PaddleOCR (and its paddlepaddle dependency) is a very heavy ML library
that we do not install for unit testing. We patch the `paddleocr` module
that PaddleOcrEngine imports lazily via importlib, so the class can be
constructed and exercised without the real engine.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from legal_rag.ingestion.models import OcrText
from legal_rag.ingestion.ocr import PaddleOcrEngine


class _FakePixmap:
    def save(self, path):
        # Simulate writing a real (empty) image file to disk.
        path.write_bytes(b"fake-png-bytes")


class _FakePage:
    def get_pixmap(self, dpi, alpha):
        return _FakePixmap()


def _install_fake_paddleocr(monkeypatch, *, predict_return):
    fake_predictor = MagicMock()
    fake_predictor.predict.return_value = predict_return

    fake_paddleocr_class = MagicMock(return_value=fake_predictor)

    import types

    fake_module = types.ModuleType("paddleocr")
    fake_module.PaddleOCR = fake_paddleocr_class
    monkeypatch.setitem(__import__("sys").modules, "paddleocr", fake_module)
    return fake_paddleocr_class, fake_predictor


def _result_with(rec_texts, rec_scores):
    result = MagicMock()
    result.json = {"res": {"rec_texts": rec_texts, "rec_scores": rec_scores}}
    return result


def test_constructor_rejects_unsupported_language(monkeypatch):
    _install_fake_paddleocr(monkeypatch, predict_return=[])
    with pytest.raises(ValueError, match="Unsupported OCR language"):
        PaddleOcrEngine(language="fr")


def test_constructor_selects_arabic_recognition_model(monkeypatch):
    fake_class, _ = _install_fake_paddleocr(monkeypatch, predict_return=[])
    PaddleOcrEngine(language="ar")
    _, kwargs = fake_class.call_args
    assert kwargs["text_recognition_model_name"] == "arabic_PP-OCRv5_mobile_rec"


def test_constructor_selects_english_recognition_model(monkeypatch):
    fake_class, _ = _install_fake_paddleocr(monkeypatch, predict_return=[])
    PaddleOcrEngine(language="en")
    _, kwargs = fake_class.call_args
    assert kwargs["text_recognition_model_name"] == "en_PP-OCRv5_mobile_rec"


def test_extract_page_returns_empty_text_when_no_results(monkeypatch):
    _install_fake_paddleocr(monkeypatch, predict_return=[])
    engine = PaddleOcrEngine(language="ar")

    result = engine.extract_page(_FakePage())

    assert result == OcrText(text="", mean_confidence=None)


def test_extract_page_joins_nonempty_recognized_lines(monkeypatch):
    result_obj = _result_with(rec_texts=["line one", "  ", "line two"], rec_scores=[0.9, 0.1, 0.8])
    _, predictor = _install_fake_paddleocr(monkeypatch, predict_return=[result_obj])
    engine = PaddleOcrEngine(language="ar")

    ocr_text = engine.extract_page(_FakePage())

    assert ocr_text.text == "line one\nline two"
    # Blank line's score is excluded from the confidence average.
    assert ocr_text.mean_confidence == pytest.approx((0.9 + 0.8) / 2)


def test_extract_page_passes_minimum_confidence_threshold(monkeypatch):
    result_obj = _result_with(rec_texts=["x"], rec_scores=[0.9])
    _, predictor = _install_fake_paddleocr(monkeypatch, predict_return=[result_obj])
    engine = PaddleOcrEngine(language="ar", minimum_confidence=0.5)

    engine.extract_page(_FakePage())

    _, kwargs = predictor.predict.call_args
    assert kwargs["text_rec_score_thresh"] == 0.5


def test_extract_page_returns_none_confidence_when_all_lines_blank(monkeypatch):
    result_obj = _result_with(rec_texts=["   ", ""], rec_scores=[0.9, 0.8])
    _install_fake_paddleocr(monkeypatch, predict_return=[result_obj])
    engine = PaddleOcrEngine(language="ar")

    ocr_text = engine.extract_page(_FakePage())

    assert ocr_text.text == ""
    assert ocr_text.mean_confidence is None
