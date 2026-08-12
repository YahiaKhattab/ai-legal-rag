# Ingestion Architecture

This is the short code tour for the extraction → OCR → normalization →
structure detection → chunking pipeline.

## Data flow

1. `cli.py` parses paths, provenance, language, and chunk limits.
2. `validation.py` checks the PDF and computes SHA-256.
3. `pipeline.py` selects native text or OCR page by page.
4. `normalization.py` derives retrieval text while originals stay immutable.
5. `structure.py` finds legal boundaries and carries headings across pages.
6. `chunking.py` splits only inside a section and enforces the E5 limit.
7. `pipeline.py` atomically writes pages, chunks, and the ingestion report.

## Module responsibilities

### `models.py`

Defines persisted contracts and contains no extraction or chunking logic.
Central contracts make future Qdrant payload changes easy to review.

- `native_text`: exact output from the PDF text layer, even when corrupt.
- `original_text`: exact selected native or OCR evidence.
- `normalized_text`: deterministic search form derived from the original.

### `validation.py`

Rejects missing, renamed, empty, oversized, encrypted, unreadable, and
zero-page PDFs. It computes the content hash used as `document_id`.

### `quality.py`

Measures script ratios and corruption indicators. The pipeline uses this
evidence to decide whether a page needs OCR.

### `native.py`

Uses PyMuPDF coordinates only to detect a known RTL digit-storage problem.
Logical native text comes from pypdf because it gave better Arabic word order
in the representative documents.

### `ocr.py`

Adapts PaddleOCR behind `OcrEngine`. Other modules do not depend directly on
PaddleOCR APIs, keeping OCR replaceable and unit-testable.

### `normalization.py`

Creates a conservative retrieval form: Unicode and whitespace cleanup, selected
Arabic character equivalence, diacritic removal, and evidence-based RTL digit
correction. It never overwrites original evidence.

### `structure.py`

Recognizes high-confidence Arabic and English parts, chapters, sections,
articles, clauses, and paragraph boundaries. When uncertain, it falls back to
a generic document section instead of inventing structure.

### `tokenization.py`

Wraps the pinned Multilingual E5 Base tokenizer behind `TokenCounter`. Tests
use a deterministic counter; production uses the exact tokenizer without
loading embedding model weights.

### `chunking.py`

Splits one detected legal section at a time. It prefers structural, sentence,
then whitespace boundaries. Defaults:

- Target: 400 tokens
- Overlap: 60 content tokens
- Hard maximum: 480 tokens including E5 special tokens and `passage: `

Every ID depends only on stable document identity, version, pipeline version,
section, page, and source offsets. Reprocessing reproduces IDs.

### `pipeline.py`

Owns orchestration, not individual algorithms. It maintains document-global
chunk indices, performs atomic writes, writes the report last, and reuses an
ingestion only when its artifacts and exact configuration are complete.

## Persisted artifacts

`*.pages.jsonl` is extraction evidence and quality diagnostics.

`*.chunks.jsonl` is the future embedding/Qdrant input and citation payload.

`*.ingestion.json` is the completion marker, configuration record, and
document-level summary.

## Why chunks stay page-bounded

Page-bounded chunks make legal citations unambiguous and preserve exact source
offsets. A long article can span pages: `structure.py` carries its title
forward, while `chunking.py` does not merge page text. Retrieval can still
return multiple chunks from the same article later.
