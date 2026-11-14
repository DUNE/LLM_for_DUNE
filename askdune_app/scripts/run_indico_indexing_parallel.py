"""Parallel Indico extraction, serial embedding/indexing, isolated benchmark DB."""

import argparse
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import chromadb

from config import EMBEDDING_MODEL, INDEXING_DB_PATH, PROD_DB_PATH
from src.analytics.index_inventory import (
    chromadb_storage_bytes,
    collection_stats,
    print_inventory_comparison,
)
from src.analytics.token_stats import count_parent_tokens, count_child_embedding_tokens
from src.embedder.embedding_wrappers import OriginalEmbedder
from src.extractors.indico_extractor_parent_child import IndicoParentChildExtractor
from src.extractors.indico_parent_builder import build_indico_parents
from src.indexing.indico_child_indexer import build_indico_child_records
from src.indexing.indico_indexer import index_indico_event

_thread_state = threading.local()


def worker_extractor():
    """Never share requests.Session-backed extractors between worker threads."""
    if not hasattr(_thread_state, "extractor"):
        extractor = IndicoParentChildExtractor()
        extractor._ensure_access()
        _thread_state.extractor = extractor
    return _thread_state.extractor


def prepare_event(event):
    event_id = str(event["id"])
    extractor = worker_extractor()
    attachments = list(extractor.iter_event_attachments(event))
    parents = []
    for attachment in attachments:
        parents.extend(build_indico_parents(attachment))
    return event_id, len(attachments), parents


def guarded_experiment_path(raw_path):
    if not raw_path:
        raise RuntimeError("--index-path is required for this experimental script")
    if not INDEXING_DB_PATH or not PROD_DB_PATH:
        raise RuntimeError("INDEXING_DB_PATH and PROD_DB_PATH must be configured")
    path = Path(raw_path).expanduser().resolve()
    protected = [Path(p).expanduser().resolve() for p in (INDEXING_DB_PATH, PROD_DB_PATH)]
    for root in protected:
        if path == root or path in root.parents or root in path.parents:
            raise RuntimeError(f"Refusing path overlapping protected database: {path} / {root}")
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f"Experimental database must be empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path

def discover_events(extractor, args):
    import re
    from urllib.parse import urljoin
    from bs4 import BeautifulSoup

    if args.event_id is not None:
        return [{"id": args.event_id}]

    import json
    from pathlib import Path

    root_category = 443
    manifest_path = Path("data/indico_events_443.json")

    if manifest_path.exists():
        with manifest_path.open() as f:
            cached_events = json.load(f)

        print(
            f"Loaded {len(cached_events)} events from {manifest_path}",
            flush=True,
        )
        return cached_events[args.start:args.start + args.limit]

    visited_categories = set()
    events_by_id = {}

    def visit_category(category_id):
            if category_id in visited_categories:
                return
    
            visited_categories.add(category_id)
    
            print(
                f"\nVisiting category {category_id} "
                f"({len(visited_categories)} categories visited)",
                flush=True,
            )
    
            # Discover events; report inaccessible categories.
            try:
                events = extractor.discover_category_events(category_id)
            except Exception as exc:
                from requests.exceptions import HTTPError

                if (
                    isinstance(exc, HTTPError)
                    and exc.response is not None
                    and exc.response.status_code == 403
                ):
                    print(
                        f"SKIPPED: Category {category_id} "
                        f"(HTTP 403 Forbidden)",
                        flush=True,
                    )
                    return

                raise

            for event in events:
                event_id = event.get("id")
                if event_id is not None:
                    events_by_id.setdefault(str(event_id), event)
    
            # Discover direct child categories only.
            url = f"{extractor.base_url}/category/{category_id}/"
            response = extractor.session.get(url, timeout=45)
            response.raise_for_status()
    
            soup = BeautifulSoup(response.text, "html.parser")
    
            children = set()
    
            for anchor in soup.select("ul.category-list a.name[href]"):
                href = anchor.get("href", "")
    
                match = re.search(
                    r"/category/(\d+)/?$",
                    href,
                )
    
                if match:
                    children.add(int(match.group(1)))
    
            for child_id in sorted(children):
                visit_category(child_id)
    
    visit_category(root_category)

    all_events = list(events_by_id.values())

    print(
        f"\nDiscovery complete: "
        f"{len(visited_categories)} categories, "
        f"{len(all_events)} unique events",
        flush=True,
    )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("w") as f:
        json.dump(all_events, f, indent=2)

    print(f"Saved event manifest: {manifest_path}", flush=True)

    return all_events[args.start:args.start + args.limit]

def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-id", type=int)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-pending", type=int, default=16)
    parser.add_argument("--index-path", required=True)
    args = parser.parse_args()

    if args.limit < 1 or args.start < 0 or args.batch_size < 1:
        parser.error("limit/batch-size must be positive; start must be nonnegative")
    if args.workers < 1 or args.max_pending < args.workers:
        parser.error("workers must be positive and max-pending >= workers")
    if args.event_id is not None and args.event_id < 1:
        parser.error("event-id must be positive")
    if EMBEDDING_MODEL != "intfloat/e5-small-v2":
        raise RuntimeError(f"Unexpected embedding model: {EMBEDDING_MODEL}")

    experiment_path = guarded_experiment_path(args.index_path)
    print(f"Experimental database: {experiment_path}", flush=True)
    client = chromadb.PersistentClient(path=str(experiment_path))
    children = client.get_or_create_collection(name="DUNE_INDICO_CHILDREN")
    before_stats = collection_stats(children, "indico")
    storage_before = chromadb_storage_bytes(experiment_path)

    # Discovery uses its own session, never shared with workers.
    discovery_extractor = IndicoParentChildExtractor()
    discovery_extractor._ensure_access()
    events = discover_events(discovery_extractor, args)
    print(f"Selected events: {len(events)}; extraction workers: {args.workers}", flush=True)

    embedder = None
    indexed = failed = empty = skipped = 0
    parent_tokens = child_tokens = attachment_count = 0
    iterator = iter(events)
    pending = {}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        while True:
            while len(pending) < args.max_pending:
                try:
                    event = next(iterator)
                except StopIteration:
                    break
                future = pool.submit(prepare_event, event)
                pending[future] = str(event["id"])

            if not pending:
                break

            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                event_id = pending.pop(future)
                try:
                    event_id, n_attachments, parents = future.result()
                    attachment_count += n_attachments
                    if not parents:
                        empty += 1
                        print(f"No extractable attachments: {event_id}", flush=True)
                        continue
                    if embedder is None:
                        embedder = OriginalEmbedder(EMBEDDING_MODEL)
                    result = index_indico_event(
                        children, event_id, parents, embedder.embed_documents,
                        target_tokens=480, batch_size=args.batch_size,
                    )
                    print(f"Indico {event_id}: {result}", flush=True)
                    if result.get("status") == "indexed":
                        indexed += 1
                        parent_tokens += count_parent_tokens(parents)
                        child_records = [
                            child for parent in parents
                            for child in build_indico_child_records(parent, target_tokens=480)
                        ]
                        child_tokens += count_child_embedding_tokens(child_records)
                    elif result.get("status") == "skipped":
                        skipped += 1
                except Exception as exc:
                    failed += 1
                    print(f"ERROR Indico {event_id}: {type(exc).__name__}: {exc}", flush=True)

    print("\n========== PARALLEL INDICO INDEXING SUMMARY ==========")
    print(f"Requested events:       {1 if args.event_id else args.limit}")
    print(f"Events checked:         {len(events)}")
    print(f"Events no attachments:  {empty}")
    print(f"Events unchanged:       {skipped}")
    print(f"Events failed:          {failed}")
    print(f"Events indexed:         {indexed}")
    print(f"Attachments extracted:  {attachment_count}")
    print(f"Parent text tokens:     {parent_tokens:,}")
    print(f"Child embedding tokens: {child_tokens:,}")
    print(f"Elapsed time:           {time.perf_counter() - started:.2f} seconds")
    print("=" * 55)
    after_stats = collection_stats(children, "indico")
    storage_after = chromadb_storage_bytes(experiment_path)
    print_inventory_comparison(
        before_stats, after_stats, "indico",
        storage_before=storage_before, storage_after=storage_after,
    )
    print("Indico experimental indexing complete")


if __name__ == "__main__":
    main()

