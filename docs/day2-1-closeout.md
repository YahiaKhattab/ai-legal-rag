# Day 2.1 Closeout

Day 2.1 finalizes the ingestion contract before embeddings and Qdrant depend on
it. Extraction, OCR, and the existing conservative normalization behavior were
preserved. The main changes are validation, provenance, legal structure,
token-safe chunking, deterministic document behavior, and reporting.

## New files

| File | Responsibility |
| --- | --- |
| `validation.py` | Validate PDF type, header, size, encryption, readability, and hash. |
| `structure.py` | Detect conservative Arabic/English legal boundaries. |
| `tokenization.py` | Count the exact pinned E5 passage tokens. |
| `test_ingestion_validation.py` | Cover valid, renamed, and oversized inputs. |
| `test_ingestion_structure.py` | Cover legal labels and cross-page heading state. |
| `test_ingestion_tokenization.py` | Prove E5 prefix and special-token counting. |
| `test_ingestion_cli.py` | Protect teammate-facing command options. |
| `docs/ingestion.md` | Explain module ownership and the end-to-end data flow. |

## Significantly updated files

| File | Update |
| --- | --- |
| `models.py` | Final page/chunk contracts and explicit original/normalized evidence. |
| `chunking.py` | Structure-aware 400/60/480 token chunking and stable IDs. |
| `pipeline.py` | Global chunk indices, idempotency, atomic outputs, and report. |
| `cli.py` | Provenance, document version, token limits, and PDF size options. |
| `pyproject.toml` | Lightweight exact-tokenizer dependency. |
| `README.md` | Current commands, artifacts, token policy, and code-tour link. |

## Acceptance evidence in this workspace

- 39 automated tests passed.
- Total branch-aware coverage: 86.43%.
- Ruff lint: passed.
- Ruff format check: passed.
- Strict mypy: passed across 28 source/test files.
- Python syntax compilation: passed.

The workspace runtime is Python 3.12, while the project intentionally requires
Python 3.11. The authoritative final gate must therefore also run in the
project's existing Windows Python 3.11.9 virtual environment.

## Required final local verification

Before declaring Day 2.1 accepted:

1. Reinstall the updated dependencies with `.[ocr,dev]`.
2. Run Ruff, mypy, and pytest in Python 3.11.9.
3. Ingest representative normal, corrupted-layer/OCR, and large legal PDFs.
4. Confirm every chunk reports at most 480 tokens.
5. Inspect Arabic headings, article numbers, originals, and normalized text.
6. Re-run a document and confirm `DUPLICATE`.
7. Copy the same PDF to another filename and confirm it reuses the document ID.

Embeddings should begin only after those checks pass.
