"""Index Indico parent-child records without version tracking."""

from collections.abc import Callable
from typing import Any

import numpy as np

from src.indexing.indico_child_indexer import build_indico_child_records


def index_indico_event(
    child_collection: Any,
    event_id: str | int,
    parents: list[dict[str, Any]],
    embed_texts: Callable,
    *,
    target_tokens: int = 480,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Index one Indico event, skipping previously completed events."""

    event_id = str(event_id)

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if not parents:
        return {"status": "empty", "indico_id": event_id}

    existing = child_collection.get(
        where={"indico_id": event_id},
        include=["metadatas"],
    )

    existing_ids = existing.get("ids") or []
    existing_metadata = existing.get("metadatas") or []

    if existing_ids and all(
        m.get("index_status") == "complete"
        for m in existing_metadata
    ):
        return {
            "status": "skipped",
            "indico_id": event_id,
            "children": len(existing_ids),
        }

    # Remove incomplete records left by an interrupted attempt.
    if existing_ids:
        child_collection.delete(ids=existing_ids)

    parent_ids = set()
    for parent in parents:
        if str(parent.get("indico_id")) != event_id:
            raise ValueError("Parent belongs to another Indico event")

        parent_id = parent.get("document_id")
        if not parent_id or parent_id in parent_ids:
            raise ValueError("Missing or duplicate parent ID")

        parent_ids.add(parent_id)

    written_ids = []
    batch = []

    def write_batch(records):
        texts = [record["cleaned_text"] for record in records]
        vectors = np.asarray(embed_texts(texts), dtype=np.float32)

        if (
            vectors.ndim != 2
            or vectors.shape[0] != len(records)
            or vectors.shape[1] == 0
            or not np.isfinite(vectors).all()
        ):
            raise ValueError("Invalid Indico embedding batch")

        ids = [record["document_id"] for record in records]
        metadatas = []

        for record in records:
            metadata = {
                key: value
                for key, value in record.items()
                if key not in ("document_id", "cleaned_text")
                and value is not None
                and isinstance(value, (str, int, float, bool))
            }
            metadata["index_status"] = "staging"
            metadatas.append(metadata)

        child_collection.add(
            ids=ids,
            documents=texts,
            embeddings=vectors.tolist(),
            metadatas=metadatas,
        )

        written_ids.extend(ids)

    try:
        for parent in parents:
            children = build_indico_child_records(
                parent,
                target_tokens=target_tokens,
            )

            for child in children:
                batch.append(child)

                if len(batch) >= batch_size:
                    write_batch(batch)
                    batch = []

        if batch:
            write_batch(batch)

        if not written_ids:
            raise RuntimeError("No Indico children were indexed")

        # Mark completion only after every batch has been written.
        for offset in range(0, len(written_ids), batch_size):
            child_collection.update(
                ids=written_ids[offset:offset + batch_size],
                metadatas=[
                    {"index_status": "complete"}
                    for _ in written_ids[offset:offset + batch_size]
                ],
            )

    except Exception:
        # Leave no partially indexed event after a failed attempt.
        if written_ids:
            child_collection.delete(ids=written_ids)
        raise

    return {
        "status": "indexed",
        "indico_id": event_id,
        "parents": len(parent_ids),
        "children": len(written_ids),
    }
