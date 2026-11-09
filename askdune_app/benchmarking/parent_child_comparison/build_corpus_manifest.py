
#!/usr/bin/env python3
"""Read-only inventory and integrity audit for the AskDUNE benchmark corpus."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import chromadb


COLLECTIONS = (
    "DUNE_DOCDB_CHILDREN",
    "DUNE_DOCDB_VERSION_MANIFEST",
    "DUNE_INDICO_CHILDREN",
)


def rows(collection, batch_size=1000):
    total = collection.count()

    for offset in range(0, total, batch_size):
        batch = collection.get(
            limit=batch_size,
            offset=offset,
            include=["metadatas"],
        )

        for record_id, metadata in zip(
            batch["ids"], batch["metadatas"]
        ):
            yield record_id, metadata or {}


def source_entry(source, source_id):
    return {
        "source": source,
        "source_id": source_id,
        "active_version": None,
        "titles": set(),
        "urls": set(),
        "attachment_ids": set(),
        "parent_ids": set(),
        "child_count": 0,
        "status_counts": Counter(),
        "parent_child_indices": defaultdict(set),
        "declared_child_counts": defaultdict(set),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--index-path",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "benchmarking/parent_child_comparison/"
            "data/corpus_manifest.json"
        ),
    )

    args = parser.parse_args()
    db_path = args.index_path.resolve()

    if db_path.name != "benchmark_200" or not db_path.is_dir():
        parser.error(
            "--index-path must point to the existing "
            "benchmark_200 directory"
        )

    client = chromadb.PersistentClient(path=str(db_path))

    available = {
        collection.name
        for collection in client.list_collections()
    }

    missing = set(COLLECTIONS) - available

    if missing:
        raise RuntimeError(
            f"Missing required collections: {sorted(missing)}"
        )

    docdb = client.get_collection("DUNE_DOCDB_CHILDREN")
    version_manifest = client.get_collection(
        "DUNE_DOCDB_VERSION_MANIFEST"
    )
    indico = client.get_collection("DUNE_INDICO_CHILDREN")

    versions = {}
    warnings = []

    # Read active DocDB versions.
    for record_id, meta in rows(version_manifest):
        doc_id = str(
            meta.get("docdb_id")
            or record_id.removeprefix("docdb_active_version_")
        )

        version = meta.get("active_version")

        if version is None:
            warnings.append(
                f"DocDB {doc_id}: missing active version"
            )

        versions[doc_id] = (
            str(version) if version is not None else None
        )

    sources = {}
    collection_counts = {}

    # Inventory both indexed collections.
    for source, collection, id_key in (
        ("docdb", docdb, "docdb_id"),
        ("indico", indico, "indico_id"),
    ):
        collection_counts[source] = collection.count()

        for child_id, meta in rows(collection):
            source_id = meta.get(id_key)

            if source_id is None:
                warnings.append(
                    f"{source} child {child_id}: missing {id_key}"
                )
                continue

            source_id = str(source_id)

            if source == "docdb":
                active = versions.get(source_id)

                if active is None:
                    warnings.append(
                        f"DocDB {source_id}: "
                        "no active version in manifest"
                    )
                    continue

                if str(meta.get("docdb_version")) != active:
                    continue

            key = (source, source_id)

            entry = sources.setdefault(
                key,
                source_entry(source, source_id),
            )

            if source == "docdb":
                entry["active_version"] = versions[source_id]

            entry["child_count"] += 1

            parent_id = meta.get("parent_id")

            if parent_id is None:
                warnings.append(
                    f"{source} child {child_id}: missing parent_id"
                )
            else:
                parent_id = str(parent_id)
                entry["parent_ids"].add(parent_id)

                index = meta.get("child_index")

                if index is not None:
                    entry["parent_child_indices"][
                        parent_id
                    ].add(str(index))

                declared = meta.get("child_count")

                if declared is not None:
                    entry["declared_child_counts"][
                        parent_id
                    ].add(str(declared))

            for title_key in (
                "title",
                "meeting_name",
                "document_title",
            ):
                if meta.get(title_key):
                    entry["titles"].add(str(meta[title_key]))

            for url_key in ("event_url", "docdb_url"):
                if meta.get(url_key):
                    entry["urls"].add(str(meta[url_key]))

            attachment = (
                meta.get("attachment_id")
                or meta.get("vector_id")
                or meta.get("attachment_url")
            )

            if attachment:
                entry["attachment_ids"].add(str(attachment))

            status = str(
                meta.get("index_status") or "not_recorded"
            )

            entry["status_counts"][status] += 1

    # Check DocDB manifest consistency.
    for doc_id in versions:
        if ("docdb", doc_id) not in sources:
            warnings.append(
                f"DocDB {doc_id}: active manifest exists "
                "but no matching child records"
            )

    output_sources = []

    for (source, source_id), entry in sorted(
        sources.items(),
        key=lambda item: (
            item[0][0],
            int(item[0][1])
            if item[0][1].isdigit()
            else item[0][1],
        ),
    ):
        for parent_id in entry["parent_ids"]:
            declared = entry[
                "declared_child_counts"
            ][parent_id]

            actual = len(
                entry["parent_child_indices"][parent_id]
            )

            if len(declared) > 1:
                warnings.append(
                    f"{source} {source_id} parent {parent_id}: "
                    f"conflicting declared child counts "
                    f"{sorted(declared)}"
                )

            elif len(declared) == 1:
                expected = int(next(iter(declared)))

                if actual and actual != expected:
                    warnings.append(
                        f"{source} {source_id} "
                        f"parent {parent_id}: "
                        f"{actual} distinct child indices "
                        f"vs declared {expected}"
                    )

        if source == "indico":
            if any(
                status != "complete"
                for status in entry["status_counts"]
            ):
                warnings.append(
                    f"Indico {source_id}: "
                    f"child statuses "
                    f"{dict(entry['status_counts'])}"
                )

        output_sources.append({
            "source": source,
            "source_id": source_id,
            "active_version": entry["active_version"],
            "titles": sorted(entry["titles"]),
            "urls": sorted(entry["urls"]),
            "attachment_ids": sorted(
                entry["attachment_ids"]
            ),
            "parent_ids": sorted(entry["parent_ids"]),
            "parent_count": len(entry["parent_ids"]),
            "child_count": entry["child_count"],
            "status_counts": dict(
                sorted(entry["status_counts"].items())
            ),
        })

    counts = dict(
        Counter(
            entry["source"]
            for entry in output_sources
        )
    )

    canonical = json.dumps(
        output_sources,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )

    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "database_path": str(db_path),
        "collections": {
            name: client.get_collection(name).count()
            for name in COLLECTIONS
        },
        "source_counts": counts,
        "source_manifest_sha256": hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest(),
        "notes": [
            "Inventory of indexed records, not proof "
            "that every remote attachment was extracted.",
            "DocDB includes only children matching "
            "the active-version manifest.",
            "URLs are stored as-is and may be "
            "Markdown-formatted.",
            "No source text or embeddings are "
            "copied into this file.",
        ],
        "warnings": sorted(set(warnings)),
        "sources": output_sources,
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    print("DocDB sources:", counts.get("docdb", 0))
    print("Indico sources:", counts.get("indico", 0))
    print("Total sources:", len(output_sources))
    print("Child records:", collection_counts)
    print("Warnings:", len(payload["warnings"]))

    for warning in payload["warnings"][:10]:
        print(" WARNING:", warning)

    print(
        "Manifest SHA256:",
        payload["source_manifest_sha256"],
    )
    print("Saved:", args.output)


if __name__ == "__main__":
    main()

