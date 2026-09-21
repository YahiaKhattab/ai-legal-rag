from pathlib import Path

import numpy as np
from qdrant_client.models import PointStruct

from legal_rag.ingestion.models import ChunkRecord
from legal_rag.vector_store.indexer import QdrantIndexer


class FakeEmbedder:
    def embed_chunks(self, chunks: list[ChunkRecord]) -> np.ndarray:
        return np.ones(
            (len(chunks), 768),
            dtype=np.float32,
        )


class FakeStore:
    def __init__(self) -> None:
        self.created = False
        self.upserted_points: list[PointStruct] = []

    def create_collection(self) -> None:
        self.created = True

    def build_point(
        self,
        *,
        chunk: ChunkRecord,
        embedding: np.ndarray,
    ) -> PointStruct:
        return PointStruct(
            id=chunk.chunk_id,
            vector=embedding.tolist(),
            payload={
                "chunk_id": chunk.chunk_id,
            },
        )

    def upsert_points(self, points: list[PointStruct]) -> None:
        self.upserted_points.extend(points)


def test_index_directory_indexes_processed_chunks(tmp_path: Path) -> None:
    chunks_file = tmp_path / "document.chunks.jsonl"

    chunks_file.write_text(
        '{"chunk_id":"chunk-1","document_id":"doc-1","document_version":1,'
        '"chunk_index":0,"original_text":"النص الأول",'
        '"normalized_text":"النص الأول","section_type":"article",'
        '"section_title":"المادة الأولى","page_start":1,"page_end":1,'
        '"source_format":"pdf","locator_type":"page","locator_start":1,'
        '"locator_end":1,"language":"ar","document_type":"law",'
        '"source":"official","source_file":"law.pdf","file_hash":"hash",'
        '"extraction_methods":["ocr"],"original_start_char":0,'
        '"original_end_char":20,"token_count":5,'
        '"tokenizer_name":"test-tokenizer","pipeline_version":"1.0.0"}\n',
        encoding="utf-8",
    )

    store = FakeStore()

    indexer = QdrantIndexer(
        store=store,
        embedder=FakeEmbedder(),
    )

    total = indexer.index_directory(tmp_path)

    assert store.created is True
    assert total == 1
    assert len(store.upserted_points) == 1
    assert store.upserted_points[0].payload is not None
    assert store.upserted_points[0].payload["chunk_id"] == "chunk-1"
