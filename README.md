# AI Legal RAG

A local, citation-oriented retrieval-augmented generation system for Arabic and
English legal and regulatory documents.

## Current scope

The repository currently implements:

- Native PDF extraction using pypdf.
- PyMuPDF rendering and right-to-left digit-coordinate analysis.
- Conditional PaddleOCR fallback for low-quality pages.
- Arabic and English text-quality measurements.
- Text normalization and safe RTL legal-number correction.
- Page-aware chunking with source metadata.
- JSONL page and chunk persistence.
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

## PDF ingestion

Place approved local PDFs under `data/input/`. This directory is ignored by Git.

Ingest one PDF:

~~~powershell
legal-rag-ingest ".\data\input\example.pdf"
~~~

Ingest every PDF in the input directory:

~~~powershell
legal-rag-ingest ".\data\input"
~~~

Process only the first N pages:

~~~powershell
legal-rag-ingest ".\data\input\example.pdf" --pages 13
~~~

Select the expected language and output directory:

~~~powershell
legal-rag-ingest ".\data\input\example.pdf" `
    --language auto `
    --output ".\data\processed"
~~~

## Generated files

For `example.pdf`, ingestion creates:

- `data/processed/example.pages.jsonl`
- `data/processed/example.chunks.jsonl`

Page records include extraction method, normalized text, quality measurements,
OCR confidence, page number, document hash, and errors.

Chunk records include source file, document hash, page number, character
offsets, extraction method, and chunk text.

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
- Input PDFs
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
git switch -c feature/short-description
~~~

Run the quality gate, commit, push the branch, and open a pull request to
`main`.
