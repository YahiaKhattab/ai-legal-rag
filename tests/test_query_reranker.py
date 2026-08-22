import numpy as np
import pytest

import legal_rag.query.reranker as reranker_module
from legal_rag.query.models import RetrievedChunk
from legal_rag.query.reranker import CrossEncoderReranker


class FakeCrossEncoder:
    def __init__(
        self,
        model_name: str,
        *,
        device: str | None = None,
        max_length: int,
    ) -> None:
        del model_name, device
        assert max_length == 512

    def predict(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        assert pairs == [("query", "first"), ("query", "second")]
        return np.array([0.1, 0.9], dtype=np.float32)


def test_reranker_orders_candidates_by_cross_encoder_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(reranker_module, "CrossEncoder", FakeCrossEncoder)
    reranker = CrossEncoderReranker("reranker-test")
    candidates = [
        RetrievedChunk("first", "doc", 0.9, "first"),
        RetrievedChunk("second", "doc", 0.8, "second"),
    ]

    result = reranker.rerank("query", candidates, top_n=2)

    assert [chunk.chunk_id for chunk in result] == ["second", "first"]
