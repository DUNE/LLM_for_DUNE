from src.analytics.token_stats import count_parent_tokens, count_child_embedding_tokens
from src.indexing.docdb_child_indexer import build_child_records
"""Index real DocDB documents into the isolated parent-child experiment."""

import argparse
import time
from pathlib import Path

import chromadb

from config import EMBEDDING_MODEL, INDEXING_DB_PATH
from src.analytics.index_inventory import chromadb_storage_bytes, collection_stats, print_inventory_comparison
from src.embedder.embedding_wrappers import OriginalEmbedder
from src.extractors.docdb_extractor_multithreaded import DocDBExtractor
from src.indexing.docdb_indexer import index_docdb_document
from src.indexing.docdb_versioned_store import get_all_active_versions


def main():
    run_start = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    if args.start < 1 or args.limit < 1 or args.batch_size < 1:
        parser.error("start, limit, and batch-size must be positive")

    if EMBEDDING_MODEL != "intfloat/e5-small-v2":
        raise RuntimeError(f"Unexpected embedding model: {EMBEDDING_MODEL}")

    stable_path = Path(INDEXING_DB_PATH).resolve()
    experiment_path = stable_path.parent / "parent_child_experiment"

    if not experiment_path.is_dir():
        raise RuntimeError(
            f"Experimental database does not exist: {experiment_path}"
        )

    if experiment_path.resolve() == stable_path:
        raise RuntimeError("Refusing to modify the stable development database")

    print("Indexing database:", experiment_path, flush=True)
    print("DocDB range:", args.start, "to", args.start + args.limit - 1)
    print("Embedding batch size:", args.batch_size, flush=True)

    client = chromadb.PersistentClient(path=str(experiment_path))
    children = client.get_collection(name="DUNE_DOCDB_CHILDREN")
    manifest = client.get_collection(name="DUNE_DOCDB_VERSION_MANIFEST")

    before_stats = collection_stats(children, "docdb")
    storage_before = chromadb_storage_bytes(experiment_path)
    active_versions = get_all_active_versions(manifest)
    embedder = OriginalEmbedder(EMBEDDING_MODEL)
    extractor = DocDBExtractor()

    processed = 0
    parent_tokens = 0
    child_tokens = 0

    for docdb_id, version, parents in extractor.iter_prepared_docdb_documents(
        start=args.start,
        limit=args.limit,
        existing_versions=active_versions,
        chunk_size=2000,
    ):
        print(
            f"Indexing DocDB {docdb_id}, version {version}, "
            f"{len(parents)} parent chunks",
            flush=True,
        )

        result = index_docdb_document(
            children,
            manifest,
            docdb_id,
            version,
            parents,
            embedder.embed_documents,
            target_tokens=480,
            batch_size=args.batch_size,
        )

        print("Result:", result, flush=True)

        if result.get("status") == "indexed":
            processed += 1
            parent_tokens += count_parent_tokens(parents)
            child_records = [
                child
                for parent in parents
                for child in build_child_records(parent, target_tokens=480)
            ]
            child_tokens += count_child_embedding_tokens(child_records)

    print("\n========== DOCDB INDEXING RUN SUMMARY ==========")
    print(f"Requested pages:        {args.limit}")
    print(f"Page ID range:          {args.start}–{args.start + args.limit - 1}")
    print(f"Pages checked:          {extractor.run_stats['pages_checked']}")
    print(f"Pages missing:          {extractor.run_stats['pages_missing']}")
    print(f"Pages no attachments:   {extractor.run_stats['pages_no_attachments']}")
    print(f"Pages unchanged:        {extractor.run_stats['pages_unchanged']}")
    print(f"Pages failed:           {extractor.run_stats['pages_failed']}")
    print(f"Pages prepared:         {extractor.run_stats['pages_prepared']}")
    print(f"Pages indexed:          {processed}")
    print(f"Parent text tokens:     {parent_tokens:,}")
    print(f"Child embedding tokens: {child_tokens:,}")
    print(f"Elapsed time:          {time.perf_counter() - run_start:.2f} seconds")
    print("=" * 55)
    after_stats = collection_stats(children, "docdb")
    storage_after = chromadb_storage_bytes(experiment_path)
    print_inventory_comparison(before_stats, after_stats, "docdb", storage_before=storage_before, storage_after=storage_after)

    print("Real DocDB indexing run complete")


if __name__ == "__main__":
    main()
