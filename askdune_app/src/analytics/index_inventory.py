from pathlib import Path
"""Read-only inventory of parent-child ChromaDB collections."""

from collections import Counter


def collection_stats(collection, source):
    """Return document, parent, child, and attachment statistics."""
    data = collection.get(include=["metadatas"])
    metadatas = data.get("metadatas") or []

    document_field = "docdb_id" if source == "docdb" else "indico_id"

    documents = set()
    parents = set()
    attachments = set()
    statuses = Counter()

    for metadata in metadatas:
        if not metadata:
            continue

        if metadata.get(document_field) is not None:
            documents.add(str(metadata[document_field]))

        if metadata.get("parent_id") is not None:
            parents.add(str(metadata["parent_id"]))

        if source == "indico" and metadata.get("attachment_id") is not None:
            attachments.add(
                (str(metadata.get("indico_id")), str(metadata["attachment_id"]))
            )

        if source == "docdb" and metadata.get("filename"):
            attachments.add((
                str(metadata.get("docdb_id")),
                str(metadata.get("docdb_version")),
                str(metadata["filename"]),
            ))

        if metadata.get("index_status"):
            statuses[str(metadata["index_status"])] += 1

    return {
        "documents": len(documents),
        "attachments": len(attachments),
        "parents": len(parents),
        "children": len(data["ids"]),
        "statuses": dict(statuses),
    }


def print_inventory_comparison(before, after, source, storage_before=None, storage_after=None):
    """Print before/after inventory and net changes."""
    print(f"\n========== {source.upper()} INDEX INVENTORY ==========")
    print(f"{'Metric':<24} {'Before':>10} {'After':>10} {'Change':>10}")

    fields = ["documents", "attachments", "parents", "children"]

    for field in fields:
        old = before[field]
        new = after[field]
        label = {
            "documents": "Pages" if source == "docdb" else "Events",
            "attachments": "Attachments",
            "parents": "Parent chunks",
            "children": "Child chunks",
        }[field]
        print(f"{label:<24} {old:>10} {new:>10} {new-old:>+10}")

    if storage_before is not None and storage_after is not None:
        mb = 1024 * 1024
        print(
            f"{'ChromaDB storage (MB)':<24}"
            f"{storage_before / mb:>10.2f}"
            f"{storage_after / mb:>11.2f}"
            f"{(storage_after - storage_before) / mb:>+10.2f}"
        )
    print("=" * 55)



def chromadb_storage_bytes(db_path):
    """Return total file size of the ChromaDB persistence directory."""
    root = Path(db_path)
    if not root.is_dir():
        return 0

    total = 0
    for item in root.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                pass
    return total
