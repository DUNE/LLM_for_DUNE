
"""Experimental parallel DocDB indexing with a single ChromaDB writer."""

import argparse
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import chromadb

from config import EMBEDDING_MODEL, INDEXING_DB_PATH, PROD_DB_PATH
from src.analytics.index_inventory import (
    chromadb_storage_bytes,
    collection_stats,
    print_inventory_comparison,
)
from src.analytics.token_stats import (
    count_parent_tokens,
    count_child_embedding_tokens,
)
from src.embedder.embedding_wrappers import OriginalEmbedder
from src.extractors.docdb_extractor_multithreaded import DocDBExtractor
from src.indexing.docdb_child_indexer import build_child_records
from src.indexing.docdb_indexer import index_docdb_document
from src.indexing.docdb_versioned_store import get_all_active_versions


_worker_local = threading.local()


def prepare_document(docid, links, metadata):
    """Prepare one complete document using a worker-local SSO session."""
    if not hasattr(_worker_local, "extractor"):
        _worker_local.extractor = DocDBExtractor()

    extractor = _worker_local.extractor
    version = int(metadata[0]["docdb_version"])

    parents = extractor.extract_complete_docdb_document(
        docid=docid,
        links=links,
        metadata=metadata,
        chunk_size=2000,
    )

    return docid, version, parents


def validate_path(path):
    """Restrict writes to a new, isolated experimental database."""
    if not path:
        raise RuntimeError("--index-path is required")

    target = Path(path).expanduser().resolve()

    protected = [
        Path(PROD_DB_PATH).expanduser().resolve(),
        Path(INDEXING_DB_PATH).expanduser().resolve(),
    ]

    for root in protected:
        if target == root or root in target.parents or target in root.parents:
            raise RuntimeError(f"Protected database path: {target}")

    if target.exists() and any(target.iterdir()):
        raise RuntimeError(
            f"Experimental database is not empty: {target}"
        )

    return target


def main():
    start_time = time.perf_counter()

    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-pending", type=int, default=8)
    parser.add_argument("--index-path", required=True)
    args = parser.parse_args()

    if min(
        args.start, args.limit, args.batch_size,
        args.workers, args.max_pending
    ) < 1:
        parser.error("All numeric arguments must be positive")

    if args.max_pending < args.workers:
        parser.error("--max-pending must be >= --workers")

    if EMBEDDING_MODEL != "intfloat/e5-small-v2":
        raise RuntimeError(f"Unexpected embedding model: {EMBEDDING_MODEL}")

    if not PROD_DB_PATH or not INDEXING_DB_PATH:
        raise RuntimeError("Database paths are not configured")

    path = validate_path(args.index_path)
    path.mkdir(parents=True, exist_ok=True)

    print(f"Experimental database: {path}", flush=True)
    print(f"DocDB range: {args.start} to {args.start + args.limit - 1}")
    print(f"Extraction workers: {args.workers}")
    print(f"Maximum pending documents: {args.max_pending}")
    print(f"Embedding batch size: {args.batch_size}", flush=True)

    client = chromadb.PersistentClient(path=str(path))
    children = client.get_or_create_collection("DUNE_DOCDB_CHILDREN")
    manifest = client.get_or_create_collection("DUNE_DOCDB_VERSION_MANIFEST")

    before_stats = collection_stats(children, "docdb")
    storage_before = chromadb_storage_bytes(path)

    active_versions = get_all_active_versions(manifest)

    embedder = OriginalEmbedder(EMBEDDING_MODEL)
    discovery = DocDBExtractor()

    processed = 0
    parent_tokens = 0
    child_tokens = 0
    pages_prepared = 0
    extraction_failures = 0
    indexing_failures = 0

    def commit(future):
        nonlocal processed, parent_tokens, child_tokens
        nonlocal pages_prepared, extraction_failures, indexing_failures

        try:
            docid, version, parents = future.result()
        except Exception as exc:
            extraction_failures += 1
            print(f"[EXTRACTION FAILED] {exc}", flush=True)
            return

        if not parents:
            extraction_failures += 1
            print(f"[EXTRACTION FAILED] DocDB {docid}", flush=True)
            return

        pages_prepared += 1

        print(
            f"Indexing DocDB {docid}, version {version}, "
            f"{len(parents)} parent chunks",
            flush=True,
        )

        try:
            result = index_docdb_document(
                children,
                manifest,
                docid,
                version,
                parents,
                embedder.embed_documents,
                target_tokens=480,
                batch_size=args.batch_size,
            )
        except Exception as exc:
            indexing_failures += 1
            print(
                f"[INDEXING FAILED] DocDB {docid}: {exc}",
                flush=True,
            )
            return

        print("Result:", result, flush=True)

        if result.get("status") in ("indexed", "updated"):
            processed += 1
            parent_tokens += count_parent_tokens(parents)

            child_records = [
                child
                for parent in parents
                for child in build_child_records(
                    parent, target_tokens=480
                )
            ]
            child_tokens += count_child_embedding_tokens(child_records)

    pending = deque()

    # Discovery remains sequential. Only document preparation is parallel.
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for docid, links, metadata in discovery.iter_changed_docdb_pages(
            start=args.start,
            limit=args.limit,
            existing_versions=active_versions,
        ):
            future = executor.submit(
                prepare_document, docid, links, metadata
            )
            pending.append(future)

            # Bound memory usage and commit in discovery order.
            if len(pending) >= args.max_pending:
                commit(pending.popleft())

        while pending:
            commit(pending.popleft())

    stats = discovery.run_stats
    total_failed = (
        stats["pages_failed"]
        + extraction_failures
        + indexing_failures
    )

    print("\n========== PARALLEL DOCDB INDEXING SUMMARY ==========")
    print(f"Requested pages:        {args.limit}")
    print(f"Page ID range:          {args.start}-{args.start + args.limit - 1}")
    print(f"Pages checked:          {stats['pages_checked']}")
    print(f"Pages missing:          {stats['pages_missing']}")
    print(f"Pages no attachments:   {stats['pages_no_attachments']}")
    print(f"Pages unchanged:        {stats['pages_unchanged']}")
    print(f"Pages failed:           {total_failed}")
    print(f"Pages prepared:         {pages_prepared}")
    print(f"Pages indexed:          {processed}")
    print(f"Parent text tokens:     {parent_tokens:,}")
    print(f"Child embedding tokens: {child_tokens:,}")
    print(f"Elapsed time:           {time.perf_counter() - start_time:.2f} seconds")
    print("=" * 55)

    after_stats = collection_stats(children, "docdb")
    storage_after = chromadb_storage_bytes(path)

    print_inventory_comparison(
        before_stats,
        after_stats,
        "docdb",
        storage_before=storage_before,
        storage_after=storage_after,
    )

    print("Experimental parallel indexing complete")


if __name__ == "__main__":
    main()
