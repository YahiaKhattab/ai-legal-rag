"""Command-line entry point for end-to-end legal document ingestion."""

from __future__ import annotations

import argparse
from pathlib import Path

from legal_rag.config import Settings
from legal_rag.embeddings.batch import BatchEmbedder
from legal_rag.embeddings.encoder import EmbeddingEncoder
from legal_rag.embeddings.models import EmbeddingConfig
from legal_rag.ingestion.models import ChunkingConfig
from legal_rag.ingestion.pipeline import IngestionPipeline
from legal_rag.ingestion.validation import DEFAULT_MAXIMUM_DOCUMENT_BYTES
from legal_rag.vector_store.indexer import QdrantIndexer
from legal_rag.vector_store.qdrant import QdrantVectorStore

SUPPORTED_SUFFIXES = frozenset({".pdf", ".docx", ".txt"})


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="legal-rag-ingest",
        description=(
            "Ingest legal documents and automatically index their chunks "
            "into the local Qdrant vector store."
        ),
    )

    parser.add_argument(
        "documents",
        nargs="+",
        type=Path,
        help="PDF, DOCX, TXT file(s), or a directory containing them.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed"),
        help="Directory where processed artifacts are stored.",
    )

    parser.add_argument(
        "--pages",
        type=int,
        default=None,
        help="Process only the first N pages (PDF only).",
    )

    parser.add_argument(
        "--language",
        choices=("ar", "en", "auto"),
        default="auto",
        help=("Expected language. Use auto for automatic Arabic/English detection."),
    )

    parser.add_argument(
        "--document-type",
        default="unknown",
    )

    parser.add_argument(
        "--source",
        default="unknown",
    )

    parser.add_argument(
        "--document-version",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--target-tokens",
        type=int,
        default=400,
    )

    parser.add_argument(
        "--overlap-tokens",
        type=int,
        default=60,
    )

    parser.add_argument(
        "--maximum-tokens",
        type=int,
        default=480,
    )

    parser.add_argument(
        "--maximum-document-mb",
        "--maximum-pdf-mb",
        dest="maximum_document_mb",
        type=int,
        default=DEFAULT_MAXIMUM_DOCUMENT_BYTES // (1024 * 1024),
        help=("Maximum size per input document (legacy --maximum-pdf-mb is also accepted)."),
    )

    return parser


def _expand_inputs(inputs: list[Path]) -> list[Path]:
    """Expand directories into supported legal document files."""

    paths: list[Path] = []

    for input_path in inputs:
        if input_path.is_dir():
            paths.extend(
                sorted(
                    path
                    for path in input_path.iterdir()
                    if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
                )
            )
        else:
            paths.append(input_path)

    return paths


def _build_indexer(settings: Settings) -> QdrantIndexer:
    """Build the configured production indexer behind a testable boundary."""

    store = QdrantVectorStore(
        url=settings.qdrant_url,
        collection_name=settings.qdrant_collection,
    )
    embedder = BatchEmbedder(
        EmbeddingEncoder(
            EmbeddingConfig(
                model_name=settings.embedding_model,
                device=settings.embedding_device,
            )
        )
    )
    return QdrantIndexer(store=store, embedder=embedder)


def main() -> int:
    arguments = _parser().parse_args()
    settings = Settings()

    documents = _expand_inputs(arguments.documents)

    if not documents:
        raise SystemExit("No supported PDF, DOCX, or TXT files found.")

    if arguments.pages is not None and any(path.suffix.lower() != ".pdf" for path in documents):
        raise SystemExit("--pages can only be used when every input is PDF.")

    # ---------------------------------------------------------
    # 1. Create the ingestion pipeline.
    # ---------------------------------------------------------
    pipeline = IngestionPipeline(
        expected_language=arguments.language,
        chunking=ChunkingConfig(
            target_tokens=arguments.target_tokens,
            overlap_tokens=arguments.overlap_tokens,
            maximum_tokens=arguments.maximum_tokens,
        ),
        maximum_document_bytes=(arguments.maximum_document_mb * 1024 * 1024),
    )

    # ---------------------------------------------------------
    # 2. Create the Qdrant indexer.
    # ---------------------------------------------------------
    indexer = _build_indexer(settings)

    # Make sure the Qdrant collection exists before indexing.
    indexer.ensure_collection()

    total_failures = 0
    total_indexed_chunks = 0

    # ---------------------------------------------------------
    # 3. Process each document.
    # ---------------------------------------------------------
    for document_path in documents:
        print()
        print("=" * 70)
        print(f"PROCESSING: {document_path}")
        print("=" * 70)

        try:
            # -------------------------------------------------
            # Stage A: Ingestion
            # -------------------------------------------------
            summary = pipeline.ingest(
                document_path,
                arguments.output,
                page_limit=arguments.pages,
                document_version=arguments.document_version,
                document_type=arguments.document_type,
                source=arguments.source,
            )

            total_failures += summary.failed_records

            print()
            print("INGESTION RESULT")
            print("-" * 70)
            print(f"Status:          {summary.status.value}")
            print(f"Document ID:     {summary.document_id[:20]}")
            print(f"Source:          {summary.source_file}")
            print(f"Records:         {summary.processed_records}")
            print(f"Direct:          {summary.direct_records}")
            print(f"OCR:             {summary.ocr_records}")
            print(f"Failed:          {summary.failed_records}")
            print(f"Chunks:          {summary.chunks}")
            print(f"Chunks file:     {summary.chunks_output}")

            # -------------------------------------------------
            # Stage B: Qdrant indexing
            # -------------------------------------------------
            #
            # If ingestion produced chunks, index ONLY this
            # document's chunks. We intentionally do NOT call
            # index_directory(), because that would re-index
            # every processed document.
            #
            if summary.chunks > 0:
                print()
                print("INDEXING INTO QDRANT...")
                print("-" * 70)

                indexed_chunks = indexer.index_file(summary.chunks_output)

                total_indexed_chunks += indexed_chunks

                print(f"Indexed chunks: {indexed_chunks}")
            else:
                print()
                print("No chunks generated; nothing was indexed into Qdrant.")

            print()
            print("DOCUMENT STATUS: READY")

        except Exception as exc:
            total_failures += 1

            print()
            print("DOCUMENT STATUS: FAILED")
            print(f"Error: {exc}")

    # ---------------------------------------------------------
    # 4. Final summary.
    # ---------------------------------------------------------
    print()
    print("=" * 70)
    print("INGESTION + QDRANT INDEXING COMPLETE")
    print("=" * 70)
    print(f"Documents processed: {len(documents)}")
    print(f"Chunks indexed:      {total_indexed_chunks}")
    print(f"Failures:            {total_failures}")
    print("=" * 70)

    return 1 if total_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
