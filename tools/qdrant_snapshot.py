"""Move a Qdrant collection from the shared cloud cluster to a local instance.

This is a two-step, standalone tool -- it does not touch the legal_rag
pipeline or its configuration. Run it once, whenever the team wants to take
the "final" indexed data offline (onto a laptop, or onto a bank server with
no internet access).

Usage:

    # 1) Pull a snapshot from Qdrant Cloud and save it to disk
    # Only from the qdrant website 
    # https://a0fa18be-1572-4844-8f50-e33620bfe7ff.eu-central-1-0.aws.cloud.qdrant.io:6333/dashboard#/collections/legal_chunks#snapshots

    
    # 2) Restore that snapshot into a local Qdrant (e.g. the project's
    #    docker-compose Qdrant running on http://localhost:6333)
    python qdrant_snapshot.py restore \
        --url http://localhost:6333 \
        --collection legal_chunks \
        --input ./legal_chunks.snapshot [Put snapshot path here]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import httpx

TIMEOUT_SECONDS = 300.0  # snapshots can be large; give it room
MAX_ATTEMPTS = 4
RETRY_DELAY_SECONDS = 5.0

# Network hiccups (connection reset during TLS handshake, timeouts, etc.)
# happen on unstable connections. Retrying a few times with a short pause
# resolves most of them without any manual intervention.
RETRYABLE_EXCEPTIONS = (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout, httpx.RemoteProtocolError)


def _with_retries(description: str, func):
    last_error: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return func()
        except RETRYABLE_EXCEPTIONS as exc:
            last_error = exc
            print(
                f"  Network hiccup on '{description}' (attempt {attempt}/{MAX_ATTEMPTS}): {exc}. "
                f"Retrying in {RETRY_DELAY_SECONDS:.0f}s..."
            )
            time.sleep(RETRY_DELAY_SECONDS)
    raise RuntimeError(f"'{description}' failed after {MAX_ATTEMPTS} attempts") from last_error


def export_snapshot(url: str, api_key: str | None, collection: str, output: Path) -> None:
    headers = {"api-key": api_key} if api_key else {}
    base = url.rstrip("/")

    with httpx.Client(timeout=TIMEOUT_SECONDS) as client:
        print(f"Requesting snapshot creation for '{collection}' on {base} ...")

        def _create():
            response = client.post(f"{base}/collections/{collection}/snapshots", headers=headers)
            response.raise_for_status()
            return response

        create_response = _with_retries("create snapshot", _create)
        snapshot_name = create_response.json()["result"]["name"]
        print(f"Snapshot created: {snapshot_name}")

        print("Downloading snapshot ...")

        def _download():
            response = client.get(
                f"{base}/collections/{collection}/snapshots/{snapshot_name}",
                headers=headers,
            )
            response.raise_for_status()
            return response

        download_response = _with_retries("download snapshot", _download)

        output.write_bytes(download_response.content)
        size_mb = output.stat().st_size / (1024 * 1024)
        print(f"Saved to {output} ({size_mb:.1f} MB)")


def restore_snapshot(url: str, api_key: str | None, collection: str, input_path: Path) -> None:
    if not input_path.exists():
        print(f"ERROR: snapshot file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    headers = {"api-key": api_key} if api_key else {}
    base = url.rstrip("/")

    with httpx.Client(timeout=TIMEOUT_SECONDS) as client:
        print(f"Uploading snapshot to local Qdrant at {base} ...")

        def _upload():
            with input_path.open("rb") as file:
                response = client.post(
                    f"{base}/collections/{collection}/snapshots/upload",
                    headers=headers,
                    files={"snapshot": (input_path.name, file, "application/octet-stream")},
                )
            response.raise_for_status()
            return response

        _with_retries("upload snapshot", _upload)
        print(f"Collection '{collection}' restored locally. It is now usable offline.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export", help="Download a snapshot from a Qdrant instance")
    export_parser.add_argument("--url", required=True, help="Qdrant Cloud cluster URL")
    export_parser.add_argument("--api-key", default=None, help="Qdrant Cloud API key")
    export_parser.add_argument("--collection", default="legal_chunks")
    export_parser.add_argument("--output", type=Path, default=Path("legal_chunks.snapshot"))

    restore_parser = subparsers.add_parser("restore", help="Upload a snapshot into a local Qdrant instance")
    restore_parser.add_argument("--url", default="http://localhost:6333", help="Local Qdrant URL")
    restore_parser.add_argument("--api-key", default=None, help="Local Qdrant API key (usually empty)")
    restore_parser.add_argument("--collection", default="legal_chunks")
    restore_parser.add_argument("--input", type=Path, default=Path("legal_chunks.snapshot"))

    args = parser.parse_args()

    if args.command == "export":
        export_snapshot(args.url, args.api_key, args.collection, args.output)
    elif args.command == "restore":
        restore_snapshot(args.url, args.api_key, args.collection, args.input)


if __name__ == "__main__":
    main()