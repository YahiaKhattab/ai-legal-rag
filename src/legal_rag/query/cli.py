from __future__ import annotations

import argparse

from legal_rag.config import Settings
from legal_rag.query.evidence_sufficiency import (
    EvidenceSufficiencyConfig,
    EvidenceSufficiencyEvaluator,
)
from legal_rag.query.models import CitedAnswer
from legal_rag.query.ollama_client import OllamaGenerationClient
from legal_rag.query.pipeline import RAGAnswerPipeline
from legal_rag.query.query_embedder import QueryEmbedder
from legal_rag.query.reranker import CrossEncoderReranker
from legal_rag.query.retriever import LegalRetriever, RetrievalFilters
from legal_rag.vector_store.qdrant import QdrantVectorStore


def _build_pipeline(
    settings: Settings,
    *,
    retrieve_top_k: int,
    rerank_top_n: int,
) -> RAGAnswerPipeline:
    """Build the configured production pipeline behind a testable boundary."""

    store = QdrantVectorStore(
        url=settings.qdrant_url,
        collection_name=settings.qdrant_collection,
    )
    retriever = LegalRetriever(
        store=store,
        embedder=QueryEmbedder(
            model_name=settings.embedding_model,
            device=settings.embedding_device,
        ),
    )
    reranker = CrossEncoderReranker(
        model_name=settings.rerank_model,
        device=settings.rerank_device,
    )
    generator = OllamaGenerationClient(
        base_url=settings.ollama_url,
        model=settings.generation_model,
        timeout_seconds=settings.generation_timeout_seconds,
    )
    sufficiency_evaluator = EvidenceSufficiencyEvaluator(
        EvidenceSufficiencyConfig(
            enabled=settings.evidence_sufficiency_enabled,
            minimum_dense_score=settings.experimental_min_dense_score,
            identifier_override_score=settings.experimental_identifier_override_score,
            minimum_rerank_score=settings.experimental_min_rerank_score,
        )
    )
    return RAGAnswerPipeline(
        retriever=retriever,
        reranker=reranker,
        generator=generator,
        sufficiency_evaluator=sufficiency_evaluator,
        retrieve_top_k=retrieve_top_k,
        rerank_top_n=rerank_top_n,
        evidence_top_n=settings.evidence_top_n,
        generation_temperature=settings.generation_temperature,
        generation_retry_count=settings.generation_retry_count,
        maximum_context_characters=settings.maximum_context_characters,
        maximum_dense_score_drop=settings.experimental_max_dense_score_drop,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the AI Legal RAG system.")

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
        default=None,
        help="Chunks fetched from Qdrant (defaults to .env setting).",
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Chunks kept after reranking (defaults to .env setting).",
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
    settings = Settings()
    retrieve_top_k = args.top_k or settings.retrieval_top_k
    rerank_top_n = args.top_n or settings.rerank_top_n

    pipeline = _build_pipeline(
        settings,
        retrieve_top_k=retrieve_top_k,
        rerank_top_n=rerank_top_n,
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

    if result.retrieval is not None:
        retrieval = result.retrieval
        print("\nRetrieval:")
        print(f"  Decision: {'sufficient' if retrieval.sufficient else 'insufficient'}")
        print(f"  Reason: {retrieval.reason}")
        print(f"  Candidates: {retrieval.candidate_count}")
        print(f"  Used chunks: {retrieval.used_chunk_count}")
        if retrieval.top_dense_score is not None:
            print(f"  Top dense score: {retrieval.top_dense_score:.4f}")
        if retrieval.top_rerank_score is not None:
            print(f"  Top rerank score: {retrieval.top_rerank_score:.4f}")
        if result.prompt_version is not None:
            print(f"  Prompt version: {result.prompt_version}")

    if result.legal_excerpts:
        print("\nLegal Evidence:")

        for excerpt in result.legal_excerpts:
            print(f"\n{excerpt.marker}")

            print(excerpt.text.strip())

            print(f"\nSource: {excerpt.source_file or excerpt.chunk_id}")

            if excerpt.page is not None:
                print(f"Page: {excerpt.page}")

            if excerpt.section_title:
                print(f"Section: {excerpt.section_title}")

    if result.citations:
        print("\nSources:")

        for citation in result.citations:
            locator_parts = []

            if citation.section_title:
                locator_parts.append(citation.section_title)

            if citation.page is not None:
                locator_parts.append(f"Page {citation.page}")

            locator = " — ".join(locator_parts) if locator_parts else ""

            print(
                f"  {citation.marker} "
                f"{citation.source_file or citation.document_id}"
                f"{' — ' + locator if locator else ''}"
            )


if __name__ == "__main__":
    main()
