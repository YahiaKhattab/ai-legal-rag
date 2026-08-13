# AI Legal RAG

A local, citation-oriented retrieval-augmented generation system for Arabic and
English legal and regulatory documents.

## Current scope

The repository currently implements:

- PDF, DOCX, and UTF-8 TXT ingestion behind format-specific extractors.
- Native PDF extraction using pypdf.
- PyMuPDF rendering and right-to-left digit-coordinate analysis.
- Conditional PaddleOCR fallback for low-quality pages.
- Arabic and English text-quality measurements.
- Text normalization and safe RTL legal-number correction.
- Conservative Arabic/English legal-structure detection.
- Token-aware chunking using the exact Multilingual E5 Base tokenizer.
- Stable document/chunk identities and duplicate reuse by SHA-256.
- Format-aware page, block, or line citations.
- Atomic JSONL source/chunk persistence plus a document ingestion report.
- Qdrant and Ollama health checks.

Embeddings, vector indexing, retrieval, reranking, prompting, and answer
generation are planned next stages.

## Requirements

- Git
- Python 3.11
- Docker Desktop with Docker Compose
- Ollama
- Windows PowerShell commands are shown below

The project requires Python `>=3.11,<3.12`.

## First-time setup

Clone the private repository and enter it:

~~~powershell
git clone <PRIVATE-REPOSITORY-URL>
cd ai-legal-rag
~~~

Create and activate a virtual environment:

~~~powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
~~~

Install development and OCR dependencies:

~~~powershell
python -m pip install -e ".[ocr,dev]"
~~~

Create the local configuration:

~~~powershell
Copy-Item .env.example .env
~~~

Never commit `.env`.

## Start local services

Start Qdrant:

~~~powershell
docker compose up -d
docker compose ps
~~~

Install the configured Ollama model:

~~~powershell
ollama pull qwen2.5:3b
~~~

Verify both services:

~~~powershell
legal-rag-health
~~~

Both Qdrant and Ollama should report `HEALTHY`.

To stop Qdrant without deleting its volume:

~~~powershell
docker compose down
~~~

## Document ingestion

Place approved local `.pdf`, `.docx`, and `.txt` files under `data/input/`. This
directory is ignored by Git. TXT input must be UTF-8.

Ingest one document with explicit provenance:

~~~powershell
legal-rag-ingest ".\data\input\example.pdf" `
    --document-type law `
    --source "Central Bank of Egypt"
~~~

Use `unknown` only when the document type or issuing source has not yet been
verified. The pipeline never guesses either value.

Ingest every supported document in the input directory:

~~~powershell
legal-rag-ingest ".\data\input"
~~~

For PDF development runs, process only the first N pages:

~~~powershell
legal-rag-ingest ".\data\input\example.pdf" --pages 13
~~~

Automatic Arabic/English detection is the default. Use `--language ar` or
`--language en` only as a troubleshooting override. Select an output directory:

~~~powershell
legal-rag-ingest ".\data\input\example.pdf" `
    --output ".\data\processed"
~~~

Detected source and chunk language is persisted as `ar`, `en`, `mixed`, or
`unknown`. Mixed DOCX/TXT and native PDF content is classified from Unicode
script evidence, and mixed source records are classified again per chunk.
When a scanned PDF has no usable text signal, the pipeline probes lightweight
Arabic and English OCR sequentially, selects the valid result, and reuses that
choice for later scan pages. An explicit language override remains useful for
unusual or heavily mixed scanned documents.

The default chunking contract is a 400-token target, 60-token overlap, and a
480-token hard maximum. The count includes the `passage: ` prefix required by
Multilingual E5 Base. The model limit is 512 tokens.

The first ingestion may download only the pinned E5 tokenizer files. Document
text is processed locally and is never sent to the model host. Later runs use
the local tokenizer cache.

Format behavior is intentionally limited:

- PDF uses native text with page-level OCR fallback when quality is poor.
- DOCX reads body paragraphs and table rows in document order; embedded image
  OCR, headers, footers, comments, and legacy `.doc` are not included.
- TXT reads UTF-8 text and preserves line ranges. Binary or legacy-encoded text
  is rejected instead of guessed.

## Generated files

Artifacts use a stable prefix derived from the full file SHA-256:

- `data/processed/document-<hash-prefix>.sources.jsonl`
- `data/processed/document-<hash-prefix>.chunks.jsonl`
- `data/processed/document-<hash-prefix>.ingestion.json`

A `--pages N` development run adds `-first-N` to the artifact prefix so a
partial test cannot overwrite the complete document outputs.

Source records distinguish:

- `native_text`: untouched extractor output.
- `original_text`: exact selected source evidence.
- `normalized_text`: retrieval-oriented text derived without mutating either
  original field.

Chunk records include the original and normalized text, legal section metadata,
source provenance, extraction methods, the exact token count, the pinned
tokenizer identity, and the pipeline version. Citation coordinates are pages
for PDF, ordered blocks for DOCX, and lines for TXT. PDF chunks also retain
`page_start` and `page_end` for compatibility.

The ingestion report records document identity, configuration, counts, and
artifact names. Reprocessing identical bytes with identical metadata and
configuration returns `DUPLICATE` and safely reuses the completed artifacts.

See [Ingestion Architecture](docs/ingestion.md) for the module-by-module code
tour and data flow.

## Engineering quality gate

Run before opening a pull request:

~~~powershell
python -m ruff check .
python -m ruff format --check .
python -m mypy
python -m pytest --cov=legal_rag --cov-report=term-missing
git diff --check
~~~

Configured test coverage must remain at or above 80%.

## Data and security

The repository intentionally excludes:

- `.env` files and secrets
- Input documents
- Extracted and processed data
- Virtual environments and caches
- Logs and local models
- Qdrant storage

Do not commit confidential documents, generated JSONL files, credentials, or
local vector data.

## Team workflow

Create a focused branch for each change:

~~~powershell
git switch main
git pull --ff-only
git switch -c yahia/short-description
~~~

Run the quality gate, commit, push the branch, and open a pull request to
`main`.
