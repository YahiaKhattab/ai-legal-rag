"""Run a query through both dense (Qdrant) and keyword (BM25) retrieval and
print a side-by-side comparison. Standalone evaluation script -- not wired
into the answer pipeline or the installed CLI entry points.

Usage:
    python -m legal_rag.query.compare_retrieval_cli "ما هي شروط تأسيس شركة؟"
    python -m legal_rag.query.compare_retrieval_cli "termination notice period" --top-k 15
"""

from __future__ import annotations

import argparse
from pathlib import Path

from legal_rag.config import Settings
from legal_rag.query.keyword_retriever import KeywordRetriever
from legal_rag.query.query_embedder import QueryEmbedder
from legal_rag.query.retriever import LegalRetriever
from legal_rag.query.retrieval_comparison import compare
from legal_rag.vector_store.qdrant import QdrantVectorStore


def _build_dense_retriever(settings: Settings) -> LegalRetriever:
    """Mirror legal_rag_search.py's _build_pipeline wiring exactly, so the
    dense side of this comparison hits the same collection with the same
    embedding model the production CLI uses -- not LegalRetriever()'s
    unconfigured defaults.
    """
    store = QdrantVectorStore(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection_name=settings.qdrant_collection,
    )

    return LegalRetriever(
        store=store,
        embedder=QueryEmbedder(
            model_name=settings.embedding_model,
            device=settings.embedding_device,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare dense vs. keyword retrieval on one query.")
    parser.add_argument("query", help="Natural-language question, Arabic or English.")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory of *.chunks.jsonl files to build the BM25 index from.",
    )
    args = parser.parse_args()

    settings = Settings()
    top_k = args.top_k or settings.retrieval_top_k

    keyword_retriever = KeywordRetriever()
    indexed_count = keyword_retriever.load(args.processed_dir)
    print(f"Loaded {indexed_count} chunks into the BM25 index from {args.processed_dir}\n")

    dense_retriever = _build_dense_retriever(settings)
    dense_results = dense_retriever.search(args.query, top_k=top_k)
    keyword_results = keyword_retriever.search(args.query, top_k=top_k)

    result = compare(args.query, dense_results, keyword_results)
    print(result.render_table())


if __name__ == "__main__":
    main()