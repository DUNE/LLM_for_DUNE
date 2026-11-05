"""Semantic retrieval over completed Indico child chunks."""

from typing import Any

from src.retrieval.metadata_filters import matches_metadata


def search_indico_children(
    child_collection: Any,
    embedder: Any,
    query: str,
    *,
    candidate_count: int = 30,
    person: str | None = None,
    title: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[dict[str, Any]]:
    """Retrieve relevant Indico children using explicit E5 query embeddings."""

    if not query.strip() or candidate_count <= 0:
        return []

    total = child_collection.count()
    if total == 0:
        return []

    query_vector = embedder.embed_query([query])[0]

    # Retrieve a broad candidate pool before applying optional metadata
    # filters. This avoids using Chroma substring matching for names.
    requested = min(total, max(candidate_count * 5, 100))

    results = []

    while True:
        response = child_collection.query(
            query_embeddings=[query_vector],
            n_results=requested,
            where={"index_status": "complete"},
            include=["documents", "metadatas", "distances"],
        )

        ids = response["ids"][0]
        documents = response["documents"][0]
        metadatas = response["metadatas"][0]
        distances = response["distances"][0]

        results = []

        for child_id, document, metadata, distance in zip(
            ids, documents, metadatas, distances
        ):
            if not matches_metadata(
                metadata,
                source="indico",
                person=person,
                title=title,
                date_from=date_from,
                date_to=date_to,
            ):
                continue

            results.append({
                **metadata,
                "document_id": child_id,
                "cleaned_text": document,
                "semantic_distance": float(distance),
                "retrieval_method": "semantic",
            })

            if len(results) >= candidate_count:
                return results

        if len(ids) < requested or requested >= total:
            return results

        requested = min(total, requested * 2)
