from src.analytics.token_stats import count_parent_tokens, count_child_embedding_tokens
from src.indexing.indico_child_indexer import build_indico_child_records
"""Index Indico attachments into the isolated parent-child experiment."""

import argparse
import time
from pathlib import Path

import chromadb

from config import EMBEDDING_MODEL, INDEXING_DB_PATH, PROD_DB_PATH

from src.analytics.index_inventory import chromadb_storage_bytes, collection_stats, print_inventory_comparison
from src.embedder.embedding_wrappers import OriginalEmbedder
from src.extractors.indico_extractor_parent_child import (
    IndicoParentChildExtractor,
)
from src.extractors.indico_parent_builder import build_indico_parents
from src.indexing.indico_indexer import index_indico_event


def main():
    run_start = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-id", type=int)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--index-path",
        type=str,
        default=None,
        help="Path to an isolated parent-child ChromaDB index",
    )
    args = parser.parse_args()

    if args.limit < 1 or args.start < 0 or args.batch_size < 1:
        parser.error("Invalid limit, start, or batch-size")

    if args.event_id is not None and args.event_id < 1:
        parser.error("event-id must be positive")

    if EMBEDDING_MODEL != "intfloat/e5-small-v2":
        raise RuntimeError(
            f"Unexpected embedding model: {EMBEDDING_MODEL}"
        )

    if not INDEXING_DB_PATH:
        raise RuntimeError("INDEXING_DB_PATH is not configured")

    if not PROD_DB_PATH:
        raise RuntimeError("PROD_DB_PATH is not configured")

    indexing_path = Path(INDEXING_DB_PATH).expanduser().resolve()
    production_path = Path(PROD_DB_PATH).expanduser().resolve()

    experiment_path = (
        Path(args.index_path).expanduser().resolve()
        if args.index_path
        else indexing_path
    )

    if experiment_path == production_path:
        raise RuntimeError(
            f"Refusing to index into PROD_DB_PATH: {experiment_path}"
        )

    experiment_path.mkdir(parents=True, exist_ok=True)

    print("Indexing database:", experiment_path, flush=True)

    client = chromadb.PersistentClient(path=str(experiment_path))
    children = client.get_or_create_collection(
        name="DUNE_INDICO_CHILDREN"
    )

    before_stats = collection_stats(children, "indico")
    storage_before = chromadb_storage_bytes(experiment_path)
    extractor = IndicoParentChildExtractor()
    extractor._ensure_access()

    if args.event_id is not None:
        details = extractor._fetch_event_details(args.event_id)
        event = {
            **details,
            "id": args.event_id,
        }
        events = [event]
    else:
        events = []
        category_ids = extractor.collect_subcategories()

        for category_id in category_ids:
            events.extend(
                extractor._fetch_category_events(
                    category_id,
                    args.limit,
                    args.start,
                )
            )

        # Avoid processing the same event from multiple categories.
        events = list({
            str(event["id"]): event
            for event in events
            if event.get("id") is not None
        }.values())


        # Exclude completed events before applying the selection limit.
        # This prevents previously indexed events from occupying batch slots.
        completed_ids = set()

        existing_records = children.get(include=["metadatas"])

        for metadata in existing_records["metadatas"]:
            if (
                metadata
                and metadata.get("indico_id") is not None
                and metadata.get("index_status") == "complete"
            ):
                completed_ids.add(str(metadata["indico_id"]))

        total_discovered = len(events)

        events = [
            event
            for event in events
            if str(event["id"]) not in completed_ids
        ]

        print(
            f"Discovered {total_discovered} unique events; "
            f"{len(events)} not yet indexed",
            flush=True,
        )

        events = events[:args.limit]

    embedder = None
    processed = 0
    parent_tokens = 0
    child_tokens = 0
    skipped_complete = 0
    no_extractable_attachments = 0
    failed_events = 0


    for event in events:
        event_id = str(event["id"])

        try:
            existing = children.get(
                where={"indico_id": event_id},
                include=["metadatas"],
            )

            if existing["ids"] and all(
                m.get("index_status") == "complete"
                for m in existing["metadatas"]
            ):
                print(f"Skipping completed event {event_id}", flush=True)
                skipped_complete += 1
                continue

            print(f"Extracting event {event_id}", flush=True)

            attachments = list(
                extractor.iter_event_attachments(event)
            )

            parents = []
            for attachment in attachments:
                parents.extend(build_indico_parents(attachment))

            if not parents:
                print(f"No extractable attachments: {event_id}", flush=True)
                no_extractable_attachments += 1
                continue

            if embedder is None:
                embedder = OriginalEmbedder(EMBEDDING_MODEL)

            result = index_indico_event(
                children,
                event_id,
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
                    for child in build_indico_child_records(
                        parent, target_tokens=480
                    )
                ]

                child_tokens += count_child_embedding_tokens(child_records)

        except Exception as exc:
            failed_events += 1
            print(
                f"ERROR indexing Indico event {event_id}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            continue

    print("\n========== INDICO INDEXING RUN SUMMARY ==========")
    print(f"Requested events:       {1 if args.event_id is not None else args.limit}")
    if args.event_id is not None:
        print(f"Event ID:               {args.event_id}")
    else:
        print(f"Event selection offset: {args.start}")
    print(f"Events checked:         {len(events)}")
    print(f"Events no attachments:  {no_extractable_attachments}")
    print(f"Events unchanged:       {skipped_complete}")
    print(f"Events failed:          {failed_events}")
    print(f"Events prepared:        {processed}")
    print(f"Events indexed:         {processed}")
    print(f"Parent text tokens:     {parent_tokens:,}")
    print(f"Child embedding tokens: {child_tokens:,}")
    print(f"Elapsed time:          {time.perf_counter() - run_start:.2f} seconds")
    print("=" * 55)

    after_stats = collection_stats(children, "indico")
    storage_after = chromadb_storage_bytes(experiment_path)
    print_inventory_comparison(before_stats, after_stats, "indico", storage_before=storage_before, storage_after=storage_after)

    print("Indico experimental indexing run complete")


if __name__ == "__main__":
    main()
