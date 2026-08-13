# Ingestion Architecture

This is the short code tour for the multi-format extraction → normalization →
structure detection → chunking pipeline. OCR is a PDF-only fallback.

## Data flow

1. `cli.py` parses paths, provenance, optional language override, and chunk limits.
2. `validation.py` identifies and validates PDF, DOCX, or UTF-8 TXT and computes SHA-256.
3. `pipeline.py` selects the matching adapter from `extractors.py`.
4. `normalization.py` derives retrieval text while originals stay immutable.
5. `structure.py` finds legal boundaries and carries headings across pages.
6. `chunking.py` splits only inside a section and enforces the E5 limit.
7. `pipeline.py` atomically writes sources, chunks, and the ingestion report.

## Module responsibilities

### `models.py`

Defines persisted contracts and contains no extraction or chunking logic.
Central contracts make future Qdrant payload changes easy to review.

- `native_text`: exact extractor output, including a corrupt PDF text layer.
- `original_text`: exact selected native, OCR, DOCX, or TXT evidence.
- `normalized_text`: deterministic search form derived from the original.

Every source and chunk also carries `source_format`, `locator_type`, and an
inclusive locator range. PDFs use pages, DOCX uses ordered blocks, and TXT uses
lines. `SourceSegment` maps continuous DOCX/TXT text back to those coordinates.

### `validation.py`

Rejects missing, renamed, empty, oversized, encrypted, malformed, or unreadable
inputs. PDF signatures/pages, DOCX package parts, and UTF-8 TXT content are
checked before extraction. It computes the content hash used as `document_id`.

### `extractors.py`

Implements the common `DocumentExtractor` contract:

- `PdfExtractor`: one record per page; native text plus conditional OCR.
- `DocxExtractor`: one continuous record with paragraph/table-row block spans.
- `TxtExtractor`: one continuous record with line spans.

Keeping adapters separate lets every format share structure detection,
normalization, chunking, persistence, and future embeddings.

### `quality.py`

Measures script ratios and corruption indicators. It deterministically labels
selected text as `ar`, `en`, `mixed`, or `unknown`; no filename or external
language service is used. The same evidence decides whether a PDF page needs OCR.

### `native.py`

Uses PyMuPDF coordinates only to detect a known RTL digit-storage problem.
Logical native text comes from pypdf because it gave better Arabic word order
in the representative documents.

### `ocr.py`

Adapts PaddleOCR behind `OcrEngine`. Other modules do not depend directly on
PaddleOCR APIs, keeping OCR replaceable and unit-testable. Automatic mode uses
native script evidence when available. For a scan without usable text, it probes
the lightweight Arabic and English recognizers sequentially, selects a valid
result using script evidence, confidence, and length, then caches that language
for later scan pages. Explicit `ar` and `en` overrides remain available.

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
format, locator type, section, and source offsets. Reprocessing reproduces IDs.
When a continuous source is bilingual, language is classified again for each
chunk so Arabic and English sections retain accurate embedding metadata.

### `pipeline.py`

Owns orchestration, not individual algorithms. It maintains document-global
chunk indices, performs atomic writes, writes the report last, and reuses an
ingestion only when its artifacts and exact configuration are complete.

## Persisted artifacts

`*.sources.jsonl` is extraction evidence, source mappings, and quality diagnostics.

`*.chunks.jsonl` is the future embedding/Qdrant input and citation payload.

`*.ingestion.json` is the completion marker, configuration record, and
document-level summary.

## Citation boundaries

PDF chunks remain page-bounded, which keeps citations unambiguous. DOCX and TXT
are processed as continuous text so chunk quality is not damaged by short
paragraphs or lines; source-segment intersections still produce exact block or
line ranges. The pipeline never invents page numbers for formats that do not
provide stable pages.
