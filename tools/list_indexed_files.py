"""List every distinct source_file value currently stored in the collection.

Useful when a lookup for a specific filename returns NOT FOUND, so you can
see exactly what is indexed instead of guessing at spelling/case.

Usage:
    python tools\list_indexed_files.py
"""

from __future__ import annotations

from legal_rag.config import Settings
from legal_rag.vector_store.qdrant import QdrantVectorStore


def main() -> None:
    settings = Settings()
    store = QdrantVectorStore(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection_name=settings.qdrant_collection,
    )

    seen: dict[str, int] = {}
    next_offset = None

    while True:
        points, next_offset = store.client.scroll(
            collection_name=store.collection_name,
            limit=200,
            offset=next_offset,
            with_payload=["source_file"],
            with_vectors=False,
        )

        for point in points:
            payload = point.payload or {}
            name = payload.get("source_file", "<missing>")
            seen[name] = seen.get(name, 0) + 1

        if next_offset is None:
            break

    if not seen:
        print("The collection has no points at all.")
        return

    print(f"Found {len(seen)} distinct source_file value(s):")
    for name, count in sorted(seen.items()):
        print(f"  {name!r}: {count} chunk(s)")


if __name__ == "__main__":
    main()