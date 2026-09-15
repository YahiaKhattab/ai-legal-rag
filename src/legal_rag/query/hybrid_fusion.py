"""Reciprocal Rank Fusion for combining independently-ranked candidate lists.

Used to merge dense (Qdrant/E5) and keyword (BM25) retrieval results before
handing the combined candidate pool to the existing cross-encoder reranker
and evidence-sufficiency gate -- see pipeline.py's integration point and
its comment on why the gate's dense-score checks deliberately do NOT see
fused scores.

RRF combines *rank positions*, not raw scores, which is exactly why it's
usable here: dense cosine scores and BM25 scores are on incomparable
scales, but "how far down each ranked list is this chunk" is directly
comparable across both.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

from legal_rag.query.models import RetrievedChunk

# Standard RRF damping constant (the value used in the original paper and
# most production hybrid-search implementations). Higher k flattens the
# influence of top ranks relative to lower ones; lower k makes rank 1
# dominate more strongly. 60 is a reasonable default to start from, not a
# tuned value for this corpus.
DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievedChunk]],
    k: int = DEFAULT_RRF_K,
) -> list[RetrievedChunk]:
    """Fuse any number of independently-ranked chunk lists into one.

    A chunk's fused score is the sum of 1/(k + rank) across every list it
    appears in (1-based rank within that list; lists it doesn't appear in
    simply don't contribute to its score). A chunk found by multiple
    retrieval strategies -- even at a modest rank in each -- can outrank
    one found strongly by only one. That's intended: agreement across
    independent methods is itself a relevance signal.

    Returned chunks carry the fused score in `.score`. This is NOT a
    cosine similarity or a BM25 score -- callers must not feed it into
    logic tuned against either of those (e.g. a raw dense-score
    threshold). Empty input lists are handled fine and simply don't
    contribute.
    """
    fused_scores: dict[str, float] = defaultdict(float)
    representative_chunk: dict[str, RetrievedChunk] = {}

    for ranked_list in ranked_lists:
        for rank, chunk in enumerate(ranked_list, start=1):
            fused_scores[chunk.chunk_id] += 1.0 / (k + rank)
            # First list to mention a chunk_id "wins" as the representative
            # object; text/metadata are identical either way since both
            # lists describe the same underlying chunk, so this only
            # affects which object's fields carry through before `.score`
            # is overwritten below.
            representative_chunk.setdefault(chunk.chunk_id, chunk)

    fused = [
        replace(representative_chunk[chunk_id], score=score)
        for chunk_id, score in fused_scores.items()
    ]
    fused.sort(key=lambda chunk: chunk.score, reverse=True)
    return fused
