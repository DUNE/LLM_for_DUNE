"""Reconstruct complete Indico parents from indexed child records."""

from typing import Any

from src.retrieval.parent_reconstruction import reconstruct_parent_context


def reconstruct_indico_parent(
    child_collection: Any,
    matched_child: dict[str, Any],
) -> dict[str, Any]:
    """Fetch and reconstruct a complete, successfully indexed Indico parent."""

    parent_id = str(matched_child.get("parent_id") or "")
    indico_id = str(matched_child.get("indico_id") or "")

    if not parent_id or not indico_id:
        raise ValueError("Missing Indico parent or event identity")

    result = child_collection.get(
        where={
            "$and": [
                {"parent_id": parent_id},
                {"indico_id": indico_id},
                {"index_status": "complete"},
            ]
        },
        include=["documents", "metadatas"],
    )

    siblings = []

    for child_id, document, metadata in zip(
        result["ids"],
        result["documents"],
        result["metadatas"],
    ):
        siblings.append({
            **metadata,
            "document_id": child_id,
            "cleaned_text": document,
        })

    if not siblings:
        raise ValueError(
            f"No completed Indico children found for parent {parent_id}"
        )

    return reconstruct_parent_context(siblings, source="indico")
