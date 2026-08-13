"""Command-line entry point for document ingestion."""

from __future__ import annotations

import argparse
from pathlib import Path

from legal_rag.ingestion.models import ChunkingConfig
from legal_rag.ingestion.pipeline import IngestionPipeline
from legal_rag.ingestion.validation import DEFAULT_MAXIMUM_DOCUMENT_BYTES

SUPPORTED_SUFFIXES = frozenset({".pdf", ".docx", ".txt"})


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="legal-rag-ingest",
        description="Extract, normalize, and chunk PDF, DOCX, and TXT legal documents.",
    )
    parser.add_argument("documents", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    parser.add_argument("--pages", type=int, default=None, help="Process only the first N pages")
    parser.add_argument(
        "--language",
        choices=("ar", "en", "auto"),
        default="auto",
        help="Detect Arabic/English automatically (ar or en remains available as an override)",
    )
    parser.add_argument("--document-type", default="unknown")
    parser.add_argument("--source", default="unknown")
    parser.add_argument("--document-version", type=int, default=1)
    parser.add_argument("--target-tokens", type=int, default=400)
    parser.add_argument("--overlap-tokens", type=int, default=60)
    parser.add_argument("--maximum-tokens", type=int, default=480)
    parser.add_argument(
        "--maximum-document-mb",
        "--maximum-pdf-mb",
        dest="maximum_document_mb",
        type=int,
        default=DEFAULT_MAXIMUM_DOCUMENT_BYTES // (1024 * 1024),
        help="Maximum size per input document (legacy --maximum-pdf-mb is also accepted)",
    )
    return parser


def _expand_inputs(inputs: list[Path]) -> list[Path]:
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


def main() -> int:
    arguments = _parser().parse_args()
    documents = _expand_inputs(arguments.documents)
    if not documents:
        raise SystemExit("No supported PDF, DOCX, or TXT files found")
    if arguments.pages is not None and any(path.suffix.lower() != ".pdf" for path in documents):
        raise SystemExit("--pages can only be used when every input is PDF")

    pipeline = IngestionPipeline(
        expected_language=arguments.language,
        chunking=ChunkingConfig(
            target_tokens=arguments.target_tokens,
            overlap_tokens=arguments.overlap_tokens,
            maximum_tokens=arguments.maximum_tokens,
        ),
        maximum_document_bytes=arguments.maximum_document_mb * 1024 * 1024,
    )
    total_failures = 0
    for document_path in documents:
        summary = pipeline.ingest(
            document_path,
            arguments.output,
            page_limit=arguments.pages,
            document_version=arguments.document_version,
            document_type=arguments.document_type,
            source=arguments.source,
        )
        total_failures += summary.failed_records
        print(
            f"{summary.status.value.upper()} {summary.source_file}: "
            f"document={summary.document_id[:12]}, records={summary.processed_records}, "
            f"direct={summary.direct_records}, ocr={summary.ocr_records}, "
            f"failed={summary.failed_records}, chunks={summary.chunks}"
        )

    return 1 if total_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
