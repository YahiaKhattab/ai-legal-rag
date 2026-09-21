"""Check whether chunks from a specific source file exist in the Qdrant collection.

Usage:
    python tools\check_file_indexed.py example.txt
"""

from __future__ import annotations

import sys

from qdrant_client import models as qmodels

from legal_rag.config import Settings
from legal_rag.vector_store.qdrant import QdrantVectorStore


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python check_file_indexed.py <source_file_name>")
        sys.exit(1)

    filename = sys.argv[1]

    settings = Settings()
    store = QdrantVectorStore(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection_name=settings.qdrant_collection,
    )

    result = store.client.scroll(
        collection_name=store.collection_name,
        scroll_filter=qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="source_file",
                    match=qmodels.MatchValue(value=filename),
                )
            ]
        ),
        limit=5,
        with_payload=True,
        with_vectors=False,
    )

    points, _ = result

    if not points:
        print(f"NOT FOUND: no chunks with source_file == '{filename}' on the cloud collection.")
        return

    print(f"FOUND: {len(points)} sample chunk(s) for '{filename}' (showing up to 5):")
    for point in points:
        payload = point.payload or {}
        print(f"  - chunk_id={payload.get('chunk_id')}  chunk_index={payload.get('chunk_index')}")


if __name__ == "__main__":
    main()