"""Document-by-document DocDB parent-child indexing orchestration."""

from collections.abc import Callable, Iterator
from typing import Any

import numpy as np

from src.indexing.docdb_child_indexer import build_child_records


def iter_embedded_docdb_batches(
    parents: list[dict[str, Any]],
    embed_texts: Callable,
    *,
    target_tokens: int = 480,
    batch_size: int = 32,
) -> Iterator[tuple[list[dict[str, Any]], np.ndarray]]:
    """Yield embedded child batches for one fully extracted DocDB document.

    Child records and embeddings are held only for the current batch.
    The caller is responsible for staging and verifying each batch.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")

    batch = []

    def embed_batch(records):
        texts = [record["cleaned_text"] for record in records]
        vectors = np.asarray(embed_texts(texts), dtype=np.float32)

        if (
            vectors.ndim != 2
            or vectors.shape[0] != len(records)
            or vectors.shape[1] == 0
            or not np.isfinite(vectors).all()
        ):
            raise ValueError("Invalid child embedding batch")

        return vectors

    for parent in parents:
        children = build_child_records(
            parent,
            target_tokens=target_tokens,
        )

        for child in children:
            batch.append(child)

            if len(batch) >= batch_size:
                yield batch, embed_batch(batch)
                batch = []

    if batch:
        yield batch, embed_batch(batch)


def index_docdb_document(
    child_collection: Any,
    manifest_collection: Any,
    docdb_id: str | int,
    version: str | int,
    parents: list[dict[str, Any]],
    embed_texts: Callable,
    *,
    target_tokens: int = 480,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Index one fully extracted DocDB document using staged batches.

    The caller must finish extracting every attachment before calling.
    Old active children remain available until activation succeeds.
    """
    from src.indexing.docdb_versioned_store import (
        get_active_version,
        validate_newer_version,
        stage_docdb_child_batch,
        record_verified_docdb_staging,
        activate_staged_docdb_version,
        cleanup_activated_docdb_version,
        discard_incomplete_docdb_staging,
    )

    docdb_id = str(docdb_id)
    version = int(version)

    previous_version = get_active_version(
        manifest_collection, docdb_id
    )

    if not validate_newer_version(version, previous_version):
        return {
            "status": "skipped",
            "docdb_id": docdb_id,
            "active_version": previous_version,
        }

    if not parents:
        raise ValueError("Cannot index an empty DocDB document")

    # Validate the entire parent set before starting any writes.
    parent_ids = set()

    for parent in parents:
        parent_id = str(parent.get("document_id", ""))

        if not parent_id.startswith(f"{docdb_id}_"):
            raise ValueError("Parent belongs to another DocDB document")

        if parent_id in parent_ids:
            raise ValueError(f"Duplicate parent ID: {parent_id}")

        if int(parent.get("docdb_version", 0)) != version:
            raise ValueError("Parent version mismatch")

        if not str(parent.get("cleaned_text", "")).strip():
            raise ValueError(f"Empty parent text: {parent_id}")

        parent_ids.add(parent_id)

    # Recover from an interrupted attempt for this document/version.
    # Completed staging is protected and requires separate recovery.
    recovered_children = discard_incomplete_docdb_staging(
        child_collection,
        manifest_collection,
        docdb_id,
        version,
    )

    staged_ids = []
    staged_children = 0

    for children, vectors in iter_embedded_docdb_batches(
        parents,
        embed_texts,
        target_tokens=target_tokens,
        batch_size=batch_size,
    ):
        ids = stage_docdb_child_batch(
            child_collection,
            manifest_collection,
            docdb_id,
            version,
            children,
            vectors,
        )

        staged_ids.extend(ids)
        staged_children += len(ids)

        # Release the current embedding batch before proceeding.
        del vectors

    if not staged_ids:
        raise RuntimeError("No child records were staged")

    record_verified_docdb_staging(
        child_collection,
        manifest_collection,
        docdb_id,
        version,
        expected_children=staged_children,
        expected_parents=len(parent_ids),
        previous_version=previous_version,
    )

    activation = activate_staged_docdb_version(
        child_collection,
        manifest_collection,
        docdb_id,
        version,
    )

    cleanup_activated_docdb_version(
        child_collection,
        manifest_collection,
        docdb_id,
        version,
    )

    return {
        "status": "updated",
        "docdb_id": docdb_id,
        "recovered_children": recovered_children,
        "previous_version": previous_version,
        "active_version": version,
        "parents": activation["parents"],
        "children": activation["children"],
    }
