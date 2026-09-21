from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from legal_rag.ingestion.ocr import PaddleOcrEngine


class FakePixmap:
    def save(self, path: Path) -> None:
        assert path.name == "page.png"


class FakePage:
    def get_pixmap(self, *, dpi: int, alpha: bool) -> FakePixmap:
        assert dpi == 300
        assert alpha is False
        return FakePixmap()


def _install_fake_paddle(
    monkeypatch: pytest.MonkeyPatch,
    result_payloads: list[dict[str, Any]],
    predictor_options: dict[str, Any],
) -> None:
    class FakePredictor:
        def __init__(self, **options: Any) -> None:
            predictor_options.update(options)

        def predict(self, image_path: str, *, text_rec_score_thresh: float) -> list[Any]:
            assert image_path.endswith("page.png")
            assert text_rec_score_thresh == 0.4
            return [SimpleNamespace(json=payload) for payload in result_payloads]

    paddle_module = SimpleNamespace(PaddleOCR=FakePredictor)
    monkeypatch.setattr(
        "legal_rag.ingestion.ocr.importlib.import_module",
        lambda name: paddle_module,
    )


def test_extracts_nonempty_lines_and_mean_confidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictor_options: dict[str, Any] = {}
    _install_fake_paddle(
        monkeypatch,
        [
            {
                "res": {
                    "rec_texts": [" السطر الاول ", "", "السطر الثاني"],
                    "rec_scores": [0.9, 0.1, 0.7],
                }
            }
        ],
        predictor_options,
    )

    engine = PaddleOcrEngine(dpi=300, minimum_confidence=0.4)
    result = engine.extract_page(FakePage())

    assert predictor_options["text_recognition_model_name"] == "arabic_PP-OCRv5_mobile_rec"
    assert predictor_options["device"] == "cpu"
    assert result.text == "السطر الاول\nالسطر الثاني"  # noqa: RUF001
    assert result.mean_confidence == pytest.approx(0.8)


def test_returns_empty_result_when_predictor_detects_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_paddle(monkeypatch, [], {})

    result = PaddleOcrEngine(dpi=300, minimum_confidence=0.4).extract_page(FakePage())

    assert result.text == ""
    assert result.mean_confidence is None


def test_rejects_unsupported_language_before_loading_paddle() -> None:
    with pytest.raises(ValueError, match="Unsupported OCR language"):
        PaddleOcrEngine(language="fr")
