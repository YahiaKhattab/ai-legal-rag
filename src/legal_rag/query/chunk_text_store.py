"""Fallback chunk-text lookup for when Qdrant payloads carry metadata only.

The documented Qdrant point payload (see section 5.2 of the technical doc)
lists chunk_id, document_id, source_file, section metadata, page/locator
fields, language, and provenance -- it does NOT explicitly list
original_text / normalized_text. Qdrant payloads are often kept metadata-only
on purpose to keep the index small.

Generation needs the actual chunk text, not just its metadata. There are two
ways to get it:
  1. Add original_text (or normalized_text) to the payload in
     QdrantVectorStore.build_point, and read it straight off the search hit.
  2. Or use this store, which re-reads the already-persisted
     `*.chunks.jsonl` artifacts by chunk_id -- no ingestion/indexing changes
     required.

LegalRetriever tries the payload first and falls back to this store when a
text field is missing, so either integration path works without code
changes elsewhere.
"""

from __future__ import annotations

from pathlib import Path

from legal_rag.embeddings.reader import read_chunks


class ChunkTextStore:
    """Lazily indexes chunk_id -> text by scanning the processed chunk
    directory once, then serves lookups from memory.
    """

    def __init__(self, processed_dir: Path) -> None:
        self._processed_dir = processed_dir
        self._index: dict[str, str] | None = None

    def _ensure_index(self) -> dict[str, str]:
        if self._index is None:
            index: dict[str, str] = {}
            for chunk_file in sorted(self._processed_dir.glob("*.chunks.jsonl")):
                for chunk in read_chunks(chunk_file):
                    text = (
                        getattr(chunk, "original_text", None)
                        or getattr(chunk, "normalized_text", "")
                        or ""
                    )
                    index[chunk.chunk_id] = text
            self._index = index
        return self._index

    def get_text(self, chunk_id: str) -> str | None:
        return self._ensure_index().get(chunk_id)
