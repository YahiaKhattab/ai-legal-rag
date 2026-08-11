"""PaddleOCR adapter isolated from the rest of the ingestion pipeline."""

from __future__ import annotations

import importlib
import tempfile
from pathlib import Path
from typing import Any, Protocol, cast

from legal_rag.ingestion.models import OcrText

_RECOGNITION_MODEL_BY_LANGUAGE = {
    "ar": "arabic_PP-OCRv5_mobile_rec",
    "en": "en_PP-OCRv5_mobile_rec",
}


class OcrEngine(Protocol):
    def extract_page(self, page: Any) -> OcrText: ...


class PaddleOcrEngine:
    def __init__(
        self,
        *,
        language: str = "ar",
        device: str = "cpu",
        dpi: int = 300,
        minimum_confidence: float = 0.35,
    ) -> None:
        try:
            recognition_model = _RECOGNITION_MODEL_BY_LANGUAGE[language]
        except KeyError:
            supported = ", ".join(sorted(_RECOGNITION_MODEL_BY_LANGUAGE))
            raise ValueError(
                f"Unsupported OCR language {language!r}; expected one of: {supported}"
            ) from None

        paddleocr_module = importlib.import_module("paddleocr")
        paddleocr_class = paddleocr_module.PaddleOCR
        self._predictor: Any = paddleocr_class(
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name=recognition_model,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device=device,
        )
        self._dpi = dpi
        self._minimum_confidence = minimum_confidence

    def extract_page(self, page: Any) -> OcrText:
        with tempfile.TemporaryDirectory(prefix="legal-rag-ocr-") as temporary_directory:
            image_path = Path(temporary_directory) / "page.png"
            pixmap = page.get_pixmap(dpi=self._dpi, alpha=False)
            pixmap.save(image_path)

            results = list(
                self._predictor.predict(
                    str(image_path),
                    text_rec_score_thresh=self._minimum_confidence,
                )
            )

        if not results:
            return OcrText(text="", mean_confidence=None)

        payload = cast(dict[str, Any], results[0].json)
        result_data = cast(dict[str, Any], payload.get("res", payload))
        texts = cast(list[str], result_data.get("rec_texts", []))
        scores = [float(score) for score in result_data.get("rec_scores", [])]
        nonempty_texts = [text.strip() for text in texts if text.strip()]
        selected_scores = [
            score for text, score in zip(texts, scores, strict=False) if text.strip()
        ]
        confidence = (
            sum(selected_scores) / len(selected_scores) if selected_scores else None
        )
        return OcrText(text="\n".join(nonempty_texts), mean_confidence=confidence)
