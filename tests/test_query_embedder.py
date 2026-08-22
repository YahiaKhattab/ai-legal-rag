import numpy as np
import pytest

import legal_rag.query.query_embedder as embedder_module
from legal_rag.query.query_embedder import QueryEmbedder


class FakeSentenceTransformer:
    def __init__(self, model_name: str, device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device
        self.encoded_text: str | None = None

    def encode(
        self,
        text: str,
        *,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
    ) -> np.ndarray:
        assert normalize_embeddings is True
        assert convert_to_numpy is True
        self.encoded_text = text
        return np.array([0.25, 0.75], dtype=np.float32)


def test_query_embedder_uses_e5_query_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(embedder_module, "SentenceTransformer", FakeSentenceTransformer)
    embedder = QueryEmbedder("e5-test", device="cpu")

    vector = embedder.encode_query("  سؤال قانوني  ")

    assert vector == [0.25, 0.75]
    assert isinstance(embedder._model, FakeSentenceTransformer)
    assert embedder._model.encoded_text == "query: سؤال قانوني"


def test_query_embedder_rejects_blank_query(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(embedder_module, "SentenceTransformer", FakeSentenceTransformer)
    embedder = QueryEmbedder("e5-test")

    with pytest.raises(ValueError, match="must not be empty"):
        embedder.encode_query("   ")
