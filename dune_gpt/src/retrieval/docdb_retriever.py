"""DocDB retrieval and active-parent reconstruction."""

from typing import Any

from src.indexing.docdb_versioned_store import get_active_version
from src.retrieval.docdb_parent_reconstruction import reconstruct_parent


def reconstruct_active_parent(
    child_collection: Any,
    manifest_collection: Any,
    matched_child: dict[str, Any],
) -> dict[str, Any]:
    """Reconstruct a parent only if its child belongs to the active version."""
    docdb_id = str(matched_child["docdb_id"])
    parent_id = str(matched_child["parent_id"])
    matched_version = int(matched_child["docdb_version"])

    active_version = get_active_version(manifest_collection, docdb_id)

    if active_version is None or matched_version != active_version:
        raise ValueError(
            f"DocDB {docdb_id}: child version {matched_version} "
            f"is not active (active version: {active_version})"
        )

    result = child_collection.get(
        where={
            "$and": [
                {"docdb_id": docdb_id},
                {"parent_id": parent_id},
                {"docdb_version": str(active_version)},
            ]
        },
        include=["documents", "metadatas"],
    )

    if not result["ids"]:
        raise ValueError(f"No active children found for parent {parent_id}")

    siblings = [
        {
            **metadata,
            "document_id": record_id,
            "cleaned_text": text,
        }
        for record_id, text, metadata in zip(
            result["ids"],
            result["documents"],
            result["metadatas"],
        )
    ]

    parent = reconstruct_parent(siblings)

    if parent["document_id"] != parent_id:
        raise ValueError("Reconstructed parent has inconsistent parent ID")

    # Verify the source child metadata directly. The reconstructed
    # parent does not necessarily expose these as top-level fields.
    for sibling in siblings:
        if (
            str(sibling["docdb_id"]) != docdb_id
            or str(sibling["parent_id"]) != parent_id
            or int(sibling["docdb_version"]) != active_version
        ):
            raise ValueError("Active child has inconsistent metadata")

    return parent



def search_docdb_parent_contexts(
    child_collection: Any,
    manifest_collection: Any,
    query: str,
    reranker: Any,
    *,
    candidate_count: int = 30,
    top_k: int = 5,
    max_references: int = 3,
    embedder: Any = None,
) -> dict[str, Any]:
    """Retrieve LLM parent contexts and distinct attachment references."""

    from src.retrieval.child_semantic_search import search_docdb_children
    from src.retrieval.child_keyword_search import search_docdb_children_bm25
    from src.retrieval.hybrid_reranker import (
        merge_child_candidates,
        rerank_child_candidates,
    )
    from src.retrieval.parent_context_selection import select_parent_contexts
    from src.retrieval.attachment_references import select_attachment_references

    if top_k <= 0 or max_references <= 0:
        raise ValueError("top_k and max_references must be positive")

    semantic = search_docdb_children(
        child_collection,
        manifest_collection,
        query,
        candidate_count=candidate_count,
        embedder=embedder,
    )

    keyword = search_docdb_children_bm25(
        child_collection,
        manifest_collection,
        query,
        candidate_count=candidate_count,
    )

    merged = merge_child_candidates(semantic, keyword)
    ranked = rerank_child_candidates(query, merged, reranker)

    # Apply optional title-based relevance signals without restricting
    # ordinary physics questions to metadata matches.
    from src.retrieval.metadata_ranking import rerank_with_metadata

    ranked = rerank_with_metadata(
        query,
        ranked,
        source="docdb",
    )

    # Reconstruct only the highest-ranked distinct parents needed
    # for the LLM. References are selected from these same contexts.
    contexts = select_parent_contexts(
        child_collection,
        manifest_collection,
        ranked,
        top_k=top_k,
    )

    # Display references only for material included in LLM context.
    references = select_attachment_references(
        contexts,
        max_references=max_references,
    )

    return {
        "contexts": contexts,
        "references": references,
    }
