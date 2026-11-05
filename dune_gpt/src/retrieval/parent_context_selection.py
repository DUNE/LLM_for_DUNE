"""Select distinct parents from reranked child candidates."""

from typing import Any

from src.retrieval.docdb_retriever import reconstruct_active_parent


def select_parent_contexts(
    child_collection: Any,
    manifest_collection: Any,
    ranked_children: list[dict[str, Any]],
    *,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Reconstruct the highest-ranked distinct active DocDB parents."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")

    selected = []
    seen_parents = set()

    for child in ranked_children:
        docdb_id = str(child["docdb_id"])
        parent_id = str(child["parent_id"])
        parent_key = (docdb_id, parent_id)

        if parent_key in seen_parents:
            continue

        # Rechecks the active version and sibling completeness.
        parent = reconstruct_active_parent(
            child_collection,
            manifest_collection,
            child,
        )

        seen_parents.add(parent_key)

        selected.append({
            "docdb_id": docdb_id,
            "parent_id": parent_id,
            "parent": parent,
            "best_child_id": child["document_id"],
            "best_child_text": child["cleaned_text"],
            "reranker_score": child["reranker_score"],
            "retrieval_methods": child.get(
                "retrieval_methods",
                [child.get("retrieval_method", "unknown")],
            ),
        })

        if len(selected) >= top_k:
            break

    return selected
