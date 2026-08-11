"""Command-line entry point for document ingestion."""

from __future__ import annotations

import argparse
from pathlib import Path

from legal_rag.ingestion.pipeline import IngestionPipeline


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="legal-rag-ingest",
        description="Extract, OCR, normalize, and chunk legal PDFs.",
    )
    parser.add_argument("pdfs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    parser.add_argument("--pages", type=int, default=None, help="Process only the first N pages")
    parser.add_argument("--language", choices=("ar", "en", "auto"), default="ar")
    return parser


def _expand_inputs(inputs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for input_path in inputs:
        if input_path.is_dir():
            paths.extend(sorted(input_path.glob("*.pdf")))
        else:
            paths.append(input_path)
    return paths


def main() -> int:
    arguments = _parser().parse_args()
    pdfs = _expand_inputs(arguments.pdfs)
    if not pdfs:
        raise SystemExit("No PDF files found")

    pipeline = IngestionPipeline(expected_language=arguments.language)
    total_failures = 0
    for pdf_path in pdfs:
        summary = pipeline.ingest(pdf_path, arguments.output, page_limit=arguments.pages)
        total_failures += summary.failed_pages
        print(
            f"DONE {summary.source_file}: pages={summary.processed_pages}, "
            f"native={summary.native_pages}, ocr={summary.ocr_pages}, "
            f"failed={summary.failed_pages}, chunks={summary.chunks}"
        )

    return 1 if total_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
