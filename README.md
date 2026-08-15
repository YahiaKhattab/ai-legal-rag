# AI Legal RAG

A local, citation-oriented retrieval-augmented generation system for Arabic and
English legal and regulatory documents.

## Current scope

The repository currently implements:

### Document ingestion

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

### Embeddings

- Multilingual E5 Base embeddings using `sentence-transformers`.
- Model: `intfloat/multilingual-e5-base`.
- Document embeddings are generated from processed chunk `normalized_text`.
- Document text is encoded using the required `passage: ` prefix.
- Embeddings use a fixed dimensionality of 768.
- Embedding vectors are stored as `float32`.
- Batch embedding preserves the original chunk order.
- Embedding generation is performed locally.

### Vector storage

- Qdrant is used as the local vector database.
- Legal chunk embeddings are stored in the `legal_chunks` collection.
- Qdrant uses cosine similarity.
- Each Qdrant point contains:
  - the embedding vector
  - the chunk identity
  - document metadata
  - legal section metadata
  - source metadata
  - citation metadata

Qdrant point IDs must be unsigned integers or UUIDs. Since the ingestion
pipeline uses string-based chunk IDs, the vector store deterministically
converts each chunk ID into a UUID before storing the point.

The original `chunk_id` is preserved in the Qdrant payload.

### Vector indexing

Processed `.chunks.jsonl` files can be indexed directly into Qdrant.

The indexer performs the following steps:

```text
Processed Chunks
      |
      v
Read .chunks.jsonl
      |
      v
Batch Embedding
      |
      v
768-dimensional vectors
      |
      v
Build Qdrant Points
      |
      v
Upsert into Qdrant
````

أيوه، الجزء محتاج **تنسيق وتنظيف بسيط** قبل ما يتحط في `README.md`، خصوصًا إن عندك تكرار في `Requirements` و`First-time setup`، وفيه بداية/نهاية Markdown غير متناسقة.

استخدم الجزء التالي **كما هو** بدل الجزء الذي أرسلته:

````markdown
## Vector Indexing

The indexer supports:

- Indexing one processed chunks file.
- Indexing all processed chunk files in a directory.
- Creating the target Qdrant collection when necessary.
- Preserving chunk order during embedding.
- Preserving chunk metadata inside the Qdrant payload.

## Current RAG Boundary

The implemented pipeline currently ends at vector storage.

```text
Input Document
      |
      v
   Ingestion
      |
      v
   Chunking
      |
      v
data/processed/*.chunks.jsonl
      |
      v
  Embeddings
      |
      v
 Qdrant Indexing
      |
      v
Qdrant `legal_chunks`
      |
      X
      |
      v
  Retrieval          <- next stage
      |
      v
  Reranking          <- later
      |
      v
  Prompting          <- later
      |
      v
 LLM Generation      <- later
````

Retrieval, reranking, prompting, and answer generation are not implemented
yet.

## Embedding and Vector Indexing

Processed chunk files can be converted into embeddings and indexed into the
local Qdrant instance.

Make sure Qdrant is running:

```powershell
docker compose up -d
docker compose ps
```

The default local Qdrant endpoint is:

```text
http://127.0.0.1:6333
```

To index all processed chunk files:

```powershell
python -c "from pathlib import Path; from legal_rag.vector_store.indexer import QdrantIndexer; from legal_rag.vector_store.qdrant import QdrantVectorStore; store=QdrantVectorStore(collection_name='legal_chunks'); indexer=QdrantIndexer(store=store); total=indexer.index_directory(Path(r'data\processed')); print('Indexed chunks:', total)"
```

The indexer creates the `legal_chunks` collection when necessary and stores
one vector point for each processed chunk.

To verify the number of indexed points:

```powershell
python -c "from legal_rag.vector_store.qdrant import QdrantVectorStore; store=QdrantVectorStore(collection_name='legal_chunks'); print('Collection:', store.collection_name); print('Points:', store.client.count(collection_name=store.collection_name, exact=True).count)"
```

### Current Vector Configuration

| Setting            | Value                           |
| ------------------ | ------------------------------- |
| Embedding model    | `intfloat/multilingual-e5-base` |
| Vector dimension   | `768`                           |
| Data type          | `float32`                       |
| Similarity         | `COSINE`                        |
| Vector database    | Qdrant                          |
| Default collection | `legal_chunks`                  |
| Default Qdrant URL | `http://127.0.0.1:6333`         |

### Verified Local Indexing Run

The current implementation has been tested against the local Qdrant instance.

One local processed-data run successfully indexed:

```text
Collection: legal_chunks
Points: 37
```

This is a development verification result, not a fixed dataset size.

## Dependencies

The embedding and vector-storage stages use the following pinned dependencies:

* `sentence-transformers==5.1.1`
* `tokenizers==0.22.2`
* `qdrant-client==1.19.0`

The complete dependency configuration is maintained in `pyproject.toml`.

Install the project and development/OCR dependencies with:

```powershell
python -m pip install -e ".[ocr,dev]"
```

## Requirements

* Git
* Python 3.11
* Docker Desktop with Docker Compose
* Ollama

Windows PowerShell commands are shown below.

The project requires Python `>=3.11,<3.12`.

Ollama is required for the planned/local LLM generation stage. The current
embedding and Qdrant indexing stages do not require Ollama.

## First-time Setup

Clone the repository and enter it:

```powershell
git clone <PRIVATE-REPOSITORY-URL>
cd ai-legal-rag
```

Create and activate a virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install development and OCR dependencies:

```powershell
python -m pip install -e ".[ocr,dev]"
```

Create the local configuration:

```powershell
Copy-Item .env.example .env
```

Never commit `.env`.

## Start Local Services

Start Qdrant:

```powershell
docker compose up -d
docker compose ps
```

Install the configured Ollama model:

```powershell
ollama pull qwen2.5:3b
```

Verify the local services:

```powershell
legal-rag-health
```

Both Qdrant and Ollama should report `HEALTHY`.

To stop Qdrant without deleting its volume:

```powershell
docker compose down
```

## Document Ingestion

Place approved local `.pdf`, `.docx`, and `.txt` files under `data/input/`.
This directory is ignored by Git. TXT input must be UTF-8.

Ingest one document with explicit provenance:

```powershell
legal-rag-ingest ".\data\input\example.pdf" `
    --document-type law `
    --source "Central Bank of Egypt"
```

Use `unknown` only when the document type or issuing source has not yet been
verified. The pipeline never guesses either value.

Ingest every supported document in the input directory:

```powershell
legal-rag-ingest ".\data\input"
```

For PDF development runs, process only the first N pages:

```powershell
legal-rag-ingest ".\data\input\example.pdf" --pages 13
```

Automatic Arabic/English detection is the default. Use `--language ar` or
`--language en` only as a troubleshooting override.

Select an output directory:

```powershell
legal-rag-ingest ".\data\input\example.pdf" `
    --output ".\data\processed"
```

Detected source and chunk language is persisted as `ar`, `en`, `mixed`, or
`unknown`.

Mixed DOCX/TXT and native PDF content is classified from Unicode script
evidence, and mixed source records are classified again per chunk.

When a scanned PDF has no usable text signal, the pipeline probes lightweight
Arabic and English OCR sequentially, selects the valid result, and reuses that
choice for later scan pages.

An explicit language override remains useful for unusual or heavily mixed
scanned documents.

The default chunking contract is:

* 400-token target
* 60-token overlap
* 480-token hard maximum

The count includes the `passage: ` prefix required by Multilingual E5 Base.
The model limit is 512 tokens.

The first ingestion may download only the pinned E5 tokenizer files. Document
text is processed locally and is never sent to the model host. Later runs use
the local tokenizer cache.

### Format Behavior

* PDF uses native text with page-level OCR fallback when quality is poor.
* DOCX reads body paragraphs and table rows in document order.
* Embedded image OCR, headers, footers, comments, and legacy `.doc` are not
  included.
* TXT reads UTF-8 text and preserves line ranges.
* Binary or legacy-encoded text is rejected instead of guessed.

## Generated Files

Artifacts use a stable prefix derived from the full file SHA-256:

* `data/processed/document-<hash-prefix>.sources.jsonl`
* `data/processed/document-<hash-prefix>.chunks.jsonl`
* `data/processed/document-<hash-prefix>.ingestion.json`

A `--pages N` development run adds `-first-N` to the artifact prefix so a
partial test cannot overwrite the complete document outputs.

Source records distinguish:

* `native_text`: untouched extractor output.
* `original_text`: exact selected source evidence.
* `normalized_text`: retrieval-oriented text derived without mutating either
  original field.

Chunk records include:

* original and normalized text
* legal section metadata
* source provenance
* extraction methods
* exact token count
* pinned tokenizer identity
* pipeline version
* citation coordinates

Citation coordinates are pages for PDF, ordered blocks for DOCX, and lines for
TXT. PDF chunks also retain `page_start` and `page_end` for compatibility.

The ingestion report records document identity, configuration, counts, and
artifact names.

Reprocessing identical bytes with identical metadata and configuration returns
`DUPLICATE` and safely reuses the completed artifacts.

See [Ingestion Architecture](docs/ingestion.md) for the module-by-module code
tour and data flow.

## Testing

The repository currently contains tests for:

* Configuration
* Document ingestion
* Extraction
* Normalization
* Chunking
* Validation
* Embedding configuration
* Embedding encoder
* Batch embedding
* Processed chunk reading
* Qdrant collection management
* Qdrant point construction
* Qdrant upsert
* Qdrant indexing
* Qdrant indexer integration

The current test suite contains 73 tests.

The latest verified quality-gate run achieved:

```text
73 passed
88.24% total coverage
mypy: no issues
ruff: all checks passed
git diff --check: passed
```

## Engineering Quality Gate

Run before opening a pull request:

```powershell
python -m ruff check .
python -m ruff format --check .
python -m mypy
python -m pytest --cov=legal_rag --cov-report=term-missing
git diff --check
```

Configured test coverage must remain at or above 80%.

## Data and Security

The repository intentionally excludes:

* `.env` files and secrets
* Input documents
* Extracted and processed data
* Virtual environments and caches
* Logs and local models
* Qdrant storage

Do not commit confidential documents, generated JSONL files, credentials, or
local vector data.

## Team Workflow

Create a focused branch for each change:

```powershell
git switch main
git pull --ff-only
git switch -c yahia/short-description
```

Run the quality gate, commit, push the branch, and open a pull request to
`main`.

Keep each change focused on one pipeline stage and update the README when a
new stage becomes operational.

````