from __future__ import annotations

import argparse

from legal_rag.query.models import CitedAnswer
from legal_rag.query.pipeline import RAGAnswerPipeline
from legal_rag.query.retriever import LegalRetriever, RetrievalFilters


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query the AI Legal RAG system."
    )

    parser.add_argument(
        "query",
        help="Natural-language question, Arabic or English.",
    )

    parser.add_argument(
        "--language",
        default="mixed",
        choices=["ar", "en", "mixed"],
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Chunks fetched from Qdrant.",
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=6,
        help="Chunks kept after reranking.",
    )

    parser.add_argument(
        "--document-type",
        default=None,
    )

    parser.add_argument(
        "--source",
        default=None,
    )

    args = parser.parse_args()

    pipeline = RAGAnswerPipeline(
        retriever=LegalRetriever(),
        retrieve_top_k=args.top_k,
        rerank_top_n=args.top_n,
        evidence_top_n=2,
    )

    filters = RetrievalFilters(
        document_type=args.document_type,
        source=args.source,
    )

    result: CitedAnswer = pipeline.answer(
        args.query,
        language=args.language,
        filters=filters,
    )

    print("\nAnswer:")
    print(result.answer_text)

    if result.legal_excerpts:
        print("\nLegal Evidence:")

        for excerpt in result.legal_excerpts:
            print(f"\n{excerpt.marker}")

            print(excerpt.text.strip())

            print(
                f"\nSource: "
                f"{excerpt.source_file or excerpt.chunk_id}"
            )

            if excerpt.page is not None:
                print(f"Page: {excerpt.page}")

            if excerpt.section_title:
                print(
                    f"Section: {excerpt.section_title}"
                )

    if result.citations:
        print("\nSources:")

        for citation in result.citations:
            locator_parts = []

            if citation.section_title:
                locator_parts.append(
                    citation.section_title
                )

            if citation.page is not None:
                locator_parts.append(
                    f"Page {citation.page}"
                )

            locator = (
                " — ".join(locator_parts)
                if locator_parts
                else ""
            )

            print(
                f"  {citation.marker} "
                f"{citation.source_file or citation.document_id}"
                f"{' — ' + locator if locator else ''}"
            )


if __name__ == "__main__":
    main()