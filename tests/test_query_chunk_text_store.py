from dataclasses import dataclass
from pathlib import Path

import pytest

import legal_rag.query.chunk_text_store as store_module
from legal_rag.query.chunk_text_store import ChunkTextStore


@dataclass
class FakeChunk:
    chunk_id: str
    original_text: str
    normalized_text: str


def test_chunk_text_store_builds_lazy_index_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks_file = tmp_path / "document.chunks.jsonl"
    chunks_file.touch()
    calls = 0

    def fake_read_chunks(path: Path) -> list[FakeChunk]:
        nonlocal calls
        assert path == chunks_file
        calls += 1
        return [FakeChunk("chunk-1", "النص الأصلي", "النص")]

    monkeypatch.setattr(store_module, "read_chunks", fake_read_chunks)
    store = ChunkTextStore(tmp_path)

    assert store.get_text("chunk-1") == "النص الأصلي"
    assert store.get_text("missing") is None
    assert calls == 1
