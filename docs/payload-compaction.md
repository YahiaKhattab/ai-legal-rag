# Optional Qdrant payload compaction

Set LEGAL_RAG_QDRANT_OMIT_NORMALIZED_TEXT=true to omit normalized_text
from newly upserted Qdrant payloads when original_text is nonblank.
The default is false.

Supported entry points: CLI ingestion and API AddNewOpinion.

Embeddings still use normalized_text. Ingestion JSONL files retain all
fields. Original text and all other payload metadata are preserved.
When original_text is blank, normalized_text is retained as a fallback.

This setting does not migrate existing points. It affects subsequent
upserts into the configured collection. Restart the API after changing
its environment settings.

Validation:
- Focused payload tests passed.
- CLI ingestion reused existing artifacts and indexed seven chunks.
- All seven cloud payloads differed only by omitted normalized_text.
- Filtered retrieval returned text without a file-store fallback.
- API upload stored seven compact points; original text remained
  accessible after the request completed.
- Synthetic sample JSON: 11,064 bytes before, 8,305 bytes after (24.9%).
  This measures serialized payload size, not database disk or RAM usage.

Experiments used separate collections:
- legal_chunks_payload_experiment_v1
- legal_chunks_payload_api_experiment_v1

The main legal_chunks collection was not migrated.
