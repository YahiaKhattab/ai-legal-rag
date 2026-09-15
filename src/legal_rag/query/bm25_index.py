"""Standalone BM25 keyword index over already-persisted chunk text.

Deliberately independent of Qdrant and the dense embedding pipeline: this
reads the same `*.chunks.jsonl` files the dense indexer already produces
(no re-chunking, no re-embedding, no Qdrant schema change) and builds an
in-memory BM25 index over them.

Tokenizer note: the normalization rules here (Arabic digit conversion,
diacritic stripping, alef/hamza/ya/ta-marbuta normalization) intentionally
mirror the private `_normalize_text` logic already used in
evidence_sufficiency.py for identifier/lexical-overlap matching, so a term
matches consistently whether the gate or this BM25 index is looking at it.
It's duplicated here rather than imported (that function is a private
module-level helper, not a shared public API) -- worth factoring into a
shared `query/text_normalization.py` module so the two can't silently
drift apart.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

from legal_rag.embeddings.reader import read_chunks
from legal_rag.ingestion.models import ChunkRecord

_STOP_WORDS = {
    "ما", "هي", "هو", "من", "في", "على", "عن", "الى", "إلى", "هذا", "هذه",
    "ذلك", "تلك", "هل", "كيف", "ماذا", "و", "او", "أو",
    "the", "what", "is", "are", "of", "in", "on", "for", "to", "and",
}

MetadataPredicate = Callable[[ChunkRecord], bool]


@dataclass(frozen=True)
class KeywordIndexEntry:
    """One indexed chunk, kept alongside its BM25 corpus position."""

    chunk: ChunkRecord


class BM25KeywordIndex:
    """In-memory BM25 index built from processed chunk files.

    Call `build_from_directory()` once at startup (or after new documents
    are ingested) to (re)build the index. Cheap enough to rebuild from
    scratch on every ingestion run at current corpus sizes -- no
    incremental-update logic here yet; add one if rebuild time becomes a
    real cost as the corpus grows.
    """

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._entries: list[KeywordIndexEntry] = []

    @property
    def size(self) -> int:
        return len(self._entries)

    def build_from_directory(self, processed_directory: Path) -> int:
        """Scan every `*.chunks.jsonl` file in `processed_directory` and
        (re)build the BM25 index from scratch. Returns the number of
        chunks indexed.
        """
        chunks: list[ChunkRecord] = []
        for chunks_file in sorted(processed_directory.glob("*.chunks.jsonl")):
            chunks.extend(read_chunks(chunks_file))

        return self.build_from_chunks(chunks)

    def build_from_chunks(self, chunks: list[ChunkRecord]) -> int:
        """(Re)build the index directly from a list of chunks -- useful for
        tests and for callers that already have chunks in memory.
        """
        self._entries = [KeywordIndexEntry(chunk=chunk) for chunk in chunks]

        if not self._entries:
            self._bm25 = None
            return 0

        tokenized_corpus = [
            _tokenize(entry.chunk.normalized_text) for entry in self._entries
        ]
        self._bm25 = BM25Okapi(tokenized_corpus)
        return len(self._entries)

    def search(
        self,
        query: str,
        top_k: int = 20,
        metadata_filter: MetadataPredicate | None = None,
    ) -> list[tuple[ChunkRecord, float]]:
        """Return up to `top_k` (chunk, bm25_score) pairs, highest score
        first, optionally restricted to chunks matching `metadata_filter`.

        BM25 scores are unbounded (not 0-1 like cosine similarity) and are
        only meaningful relative to each other within one query -- do not
        compare raw BM25 scores across different queries.
        """
        if self._bm25 is None or not self._entries:
            return []

        tokenized_query = _tokenize(query)
        if not tokenized_query:
            return []

        scores = self._bm25.get_scores(tokenized_query)

        candidate_indices = [
            i
            for i in range(len(scores))
            if scores[i] > 0  # exclude zero-overlap chunks rather than padding results
            and (metadata_filter is None or metadata_filter(self._entries[i].chunk))
        ]
        candidate_indices.sort(key=lambda i: scores[i], reverse=True)
        top_indices = candidate_indices[:top_k]

        return [(self._entries[i].chunk, float(scores[i])) for i in top_indices]


def _tokenize(text: str) -> list[str]:
    """Normalize and tokenize text for BM25 indexing/querying.

    Applied identically to both document text (at index time) and query
    text (at search time) -- BM25 term matching only works if both sides
    normalize the same way.
    """
    normalized = _normalize_text(text)
    return [
        token
        for token in normalized.split()
        if len(token) >= 2 and token not in _STOP_WORDS
    ]


def _normalize_text(text: str) -> str:
    text = _normalize_digits(text)

    # Remove Arabic diacritics.
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)

    # Normalize common Arabic OCR/spelling variants.
    text = text.replace("أ", "ا")
    text = text.replace("إ", "ا")
    text = text.replace("آ", "ا")
    text = text.replace("ى", "ي")
    text = text.replace("ة", "ه")

    # Normalize punctuation to spaces.
    text = re.sub(r"[\(\)\[\]\{\}:،,؛;.!؟?\"'«»ـ\-_/\\]+", " ", text)

    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def _normalize_digits(value: str) -> str:
    return value.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
