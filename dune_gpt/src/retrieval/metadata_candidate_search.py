"""Retrieve metadata-matching children directly from experimental collections."""

from typing import Any

from src.indexing.docdb_versioned_store import get_all_active_versions
from src.retrieval.metadata_filters import matches_metadata


def search_metadata_children(
    collection: Any,
    *,
    source: str,
    manifest_collection: Any = None,
    person: str | None = None,
    title: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    batch_size: int = 500,
) -> list[dict[str, Any]]:
    """Find all matching children without depending on semantic relevance."""

    if source not in ("docdb", "indico"):
        raise ValueError(f"Unsupported source: {source}")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if not any(v is not None for v in (
        person, title, date_from, date_to
    )):
        return []

    if source == "docdb":
        if manifest_collection is None:
            raise ValueError("DocDB requires an active-version manifest")
        active_versions = get_all_active_versions(manifest_collection)
    else:
        active_versions = {}

    matches = []
    offset = 0

    while True:
        batch = collection.get(
            limit=batch_size,
            offset=offset,
            include=["documents", "metadatas"],
        )

        for record_id, document, metadata in zip(
            batch["ids"],
            batch["documents"],
            batch["metadatas"],
        ):
            if source == "docdb":
                docdb_id = str(metadata.get("docdb_id", ""))
                version = int(metadata.get("docdb_version", -1))

                if active_versions.get(docdb_id) != version:
                    continue

            if not matches_metadata(
                metadata,
                source=source,
                person=person,
                title=title,
                date_from=date_from,
                date_to=date_to,
            ):
                continue

            matches.append({
                **metadata,
                "source": source,
                "document_id": record_id,
                "cleaned_text": document,
                "retrieval_method": "metadata",
            })

        if len(batch["ids"]) < batch_size:
            break

        offset += batch_size

    return matches
