# AI Legal RAG — Unit Test Suite

Unit test suite for the `legal_rag` Python package (ingestion, embeddings,
vector_store, and query pipelines).

## Summary

| Metric | Value |
|---|---|
| Test framework | pytest 9.1.1 |
| Total tests | 251 |
| Result | 251 passed, 0 failed |
| Code coverage | 95% (1,543 statements, 77 missed) |
| Source files covered | 34 / 34 |

Full per-test output is in `TEST_OUTPUT.txt`. Full per-file coverage
breakdown is in `COVERAGE_REPORT.txt`.

## Project layout

Place this folder as a sibling of the `ai-legal-rag-main` project:

```
project-root/
├── ai-legal-rag-main/     (project source)
└── unit_tests/            (this folder)
```

## Setup

```bash
python -m venv .venv-tests
source .venv-tests/bin/activate        # Windows: .venv-tests\Scripts\Activate.ps1
pip install -r unit_tests/requirements-test.txt
```

## Running the tests

```bash
cd unit_tests

pytest -q                                          # summary run
pytest -v                                          # per-test output
pytest --cov=legal_rag --cov-report=term-missing   # with coverage
```

## Structure

Each test file targets one source module (`test_<module>.py` covers
`legal_rag/<module>.py`):

| Module | Test file |
|---|---|
| config.py | test_config.py |
| health.py | test_health.py |
| ingestion/models.py | test_ingestion_models.py |
| ingestion/validation.py | test_ingestion_validation.py |
| ingestion/quality.py | test_ingestion_quality.py |
| ingestion/normalization.py | test_ingestion_normalization.py |
| ingestion/structure.py | test_ingestion_structure.py |
| ingestion/tokenization.py | test_ingestion_tokenization.py |
| ingestion/chunking.py | test_ingestion_chunking.py |
| ingestion/native.py | test_ingestion_native.py |
| ingestion/ocr.py | test_ingestion_ocr.py |
| ingestion/extractors.py | test_ingestion_extractors.py |
| ingestion/pipeline.py | test_ingestion_pipeline.py |
| ingestion/cli.py | test_ingestion_cli.py |
| embeddings/models.py | test_embeddings_models.py |
| embeddings/encoder.py | test_embeddings_encoder.py |
| embeddings/batch.py | test_embeddings_batch.py |
| embeddings/reader.py | test_embeddings_reader.py |
| vector_store/qdrant.py | test_vector_store_qdrant.py |
| vector_store/indexer.py | test_vector_store_indexer.py |
| query/models.py | test_query_models.py |
| query/chunk_text_store.py | test_query_chunk_text_store.py |
| query/query_embedder.py | test_query_query_embedder.py |
| query/ollama_client.py | test_query_ollama_client.py |
| query/retriever.py | test_query_retriever.py |
| query/reranker.py | test_query_reranker.py |
| query/prompt_builder.py | test_query_prompt_builder.py |
| query/answer_validator.py | test_query_answer_validator.py |
| query/pipeline.py | test_query_pipeline.py |
| query/cli.py | test_query_cli.py |

## Approach

External services and heavy ML dependencies are not required to run this
suite:

- **Qdrant** — replaced with `unittest.mock.MagicMock` (no live server needed).
- **sentence-transformers / PaddleOCR** — `conftest.py` installs lightweight
  stub modules automatically at test-collection time; individual tests patch
  the specific classes they exercise.
- **Ollama / health-check HTTP calls** — replaced with fake `httpx` client and
  response objects to control success, failure, and malformed-response cases.
- **PDF/DOCX/TXT parsing** — exercised against real, small files generated at
  test time (via `pypdf`, `python-docx`, `pymupdf`), since this code is
  fundamentally about parsing real file formats.

## Notes

- This suite is additive to any existing integration tests in the project's
  own `tests/` directory and does not modify or replace them.
- Coverage gaps below 100% are concentrated in a small number of rare error
  branches (documented per-file in `COVERAGE_REPORT.txt`) and a CLI
  `__main__` entry point that performs live network calls.
