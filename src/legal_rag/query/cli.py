"""`legal-rag-query` CLI: one-shot grounded legal Q&A.

Mirrors the existing `legal-rag-ingest` / `legal-rag-health` CLI style so
the team has one consistent way of invoking every stage.
"""
from __future__ import annotations

import argparse

from legal_rag.query.pipeline import RAGAnswerPipeline
from legal_rag.query.retriever import LegalRetriever, RetrievalFilters


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the AI Legal RAG system.")
    parser.add_argument("query", help="Natural-language question, Arabic or English.")
    parser.add_argument("--language", default="mixed", choices=["ar", "en", "mixed"])
    parser.add_argument("--top-k", type=int, default=20, help="Chunks fetched from Qdrant.")
    parser.add_argument("--top-n", type=int, default=6, help="Chunks kept after reranking.")
    parser.add_argument("--document-type", default=None)
    parser.add_argument("--source", default=None)
    args = parser.parse_args()

    pipeline = RAGAnswerPipeline(
        retriever=LegalRetriever(),
        retrieve_top_k=args.top_k,
        rerank_top_n=args.top_n,
    )
    filters = RetrievalFilters(document_type=args.document_type, source=args.source)
    result = pipeline.answer(args.query, language=args.language, filters=filters)

    print(result.answer_text)
    print("\nSources:")
    for citation in result.citations:
        locator = citation.section_title or (f"p.{citation.page}" if citation.page else "")
        print(f"  {citation.marker} {citation.source_file} {locator}".rstrip())


if __name__ == "__main__":
    main()
