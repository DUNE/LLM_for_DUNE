"""Semantic retrieval over active DocDB child chunks."""

from typing import Any

from src.indexing.docdb_versioned_store import get_all_active_versions


def search_docdb_children(
    child_collection: Any,
    manifest_collection: Any,
    query: str,
    *,
    candidate_count: int = 30,
    embedder: Any = None,
) -> list[dict[str, Any]]:
    """Retrieve semantic child candidates belonging to active versions."""

    if not query or not query.strip():
        raise ValueError("Search query cannot be empty")

    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")

    active_versions = get_all_active_versions(manifest_collection)

    if not active_versions:
        return []

    total_children = child_collection.count()

    if total_children == 0:
        return []

    # Request additional candidates because the collection may contain
    # inactive versions or incomplete staged children.
    requested = min(total_children, max(candidate_count * 4, candidate_count))

    candidates = []
    seen = set()

    query_input = (
        {"query_embeddings": [embedder.embed_query([query])[0]]}
        if embedder is not None
        else {"query_texts": [query]}
    )

    while True:
        results = child_collection.query(
            **query_input,
            n_results=requested,
            include=["documents", "metadatas", "distances"],
        )

        for record_id, text, metadata, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            if record_id in seen:
                continue

            seen.add(record_id)

            docdb_id = str(metadata.get("docdb_id", ""))
            version = int(metadata.get("docdb_version", -1))

            if active_versions.get(docdb_id) != version:
                continue

            candidates.append({
                **metadata,
                "document_id": record_id,
                "cleaned_text": text,
                "semantic_distance": float(distance),
                "retrieval_method": "semantic",
            })

            if len(candidates) >= candidate_count:
                return candidates

        if requested >= total_children:
            break

        requested = min(total_children, requested * 2)

    return candidates
