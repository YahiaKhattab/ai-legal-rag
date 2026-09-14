"""Compare dense (Qdrant/E5) retrieval against keyword (BM25) retrieval on
the same query, side by side.

This is an evaluation tool, not part of the answer pipeline. Its purpose
is exactly what was asked for: run both retrieval strategies independently
on the same query, see what each one got, and measure how much they agree
-- *before* deciding how (or whether) to fuse them.
"""

from __future__ import annotations

from dataclasses import dataclass

from legal_rag.query.models import RetrievedChunk


@dataclass(frozen=True)
class RankedHit:
    """One retrieved chunk plus its 1-based rank within its own list."""

    rank: int
    chunk: RetrievedChunk


@dataclass(frozen=True)
class ComparisonResult:
    query: str
    dense_hits: list[RankedHit]
    keyword_hits: list[RankedHit]

    @property
    def dense_chunk_ids(self) -> set[str]:
        return {hit.chunk.chunk_id for hit in self.dense_hits}

    @property
    def keyword_chunk_ids(self) -> set[str]:
        return {hit.chunk.chunk_id for hit in self.keyword_hits}

    @property
    def overlap_chunk_ids(self) -> set[str]:
        return self.dense_chunk_ids & self.keyword_chunk_ids

    @property
    def dense_only_chunk_ids(self) -> set[str]:
        return self.dense_chunk_ids - self.keyword_chunk_ids

    @property
    def keyword_only_chunk_ids(self) -> set[str]:
        return self.keyword_chunk_ids - self.dense_chunk_ids

    @property
    def jaccard_overlap(self) -> float:
        """Size of the intersection over the size of the union, 0..1.

        A useful single number to track across many queries, but it treats
        rank position as irrelevant -- a chunk ranked #1 by both methods
        counts the same as one ranked #1 by one and #20 by the other. Use
        `render_table()` when rank position itself matters for a specific
        query.
        """
        union = self.dense_chunk_ids | self.keyword_chunk_ids
        if not union:
            return 0.0
        return len(self.overlap_chunk_ids) / len(union)

    def render_table(self) -> str:
        """Render a readable side-by-side comparison for manual review.

        Prints the full chunk_id (not truncated) since it's the value
        you'd copy to look a chunk up elsewhere -- a partial ID isn't
        useful for that.
        """
        lines = [f"Query: {self.query!r}", ""]

        lines.append(
            f"Dense hits: {len(self.dense_hits)}  |  "
            f"Keyword hits: {len(self.keyword_hits)}  |  "
            f"Overlap: {len(self.overlap_chunk_ids)}  |  "
            f"Jaccard: {self.jaccard_overlap:.2f}"
        )
        lines.append("")

        dense_by_id = {hit.chunk.chunk_id: hit for hit in self.dense_hits}
        keyword_by_id = {hit.chunk.chunk_id: hit for hit in self.keyword_hits}
        all_ids = self.dense_chunk_ids | self.keyword_chunk_ids

        # Sort by best (lowest) rank across either list, so the chunks
        # either method liked most appear first.
        def sort_key(chunk_id: str) -> int:
            dense_rank = dense_by_id[chunk_id].rank if chunk_id in dense_by_id else 999
            keyword_rank = keyword_by_id[chunk_id].rank if chunk_id in keyword_by_id else 999
            return min(dense_rank, keyword_rank)

        lines.append(f"{'dense#':>7}  {'kw#':>4}  chunk_id")
        lines.append("-" * 40)

        for chunk_id in sorted(all_ids, key=sort_key):
            dense_hit = dense_by_id.get(chunk_id)
            keyword_hit = keyword_by_id.get(chunk_id)

            dense_rank_str = str(dense_hit.rank) if dense_hit else "-"
            keyword_rank_str = str(keyword_hit.rank) if keyword_hit else "-"

            lines.append(f"{dense_rank_str:>7}  {keyword_rank_str:>4}  {chunk_id}")

        return "\n".join(lines)


def compare(
    query: str,
    dense_chunks: list[RetrievedChunk],
    keyword_chunks: list[RetrievedChunk],
) -> ComparisonResult:
    """Wrap two already-retrieved result lists for comparison.

    Kept as a plain function over two chunk lists (rather than owning the
    retrievers itself) so it works the same whether the caller ran
    LegalRetriever/KeywordRetriever live, or is replaying saved results
    from a batch evaluation run.
    """
    dense_hits = [RankedHit(rank=i + 1, chunk=chunk) for i, chunk in enumerate(dense_chunks)]
    keyword_hits = [RankedHit(rank=i + 1, chunk=chunk) for i, chunk in enumerate(keyword_chunks)]
    return ComparisonResult(query=query, dense_hits=dense_hits, keyword_hits=keyword_hits)