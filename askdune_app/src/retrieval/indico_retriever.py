"""Hybrid Indico retrieval with reranking and parent reconstruction."""

from typing import Any

from src.retrieval.indico_semantic_search import search_indico_children
from src.retrieval.indico_keyword_search import search_indico_children_bm25
from src.retrieval.indico_parent_reconstruction import reconstruct_indico_parent
from src.retrieval.hybrid_reranker import (
    merge_child_candidates,
    rerank_child_candidates,
)
from src.retrieval.metadata_filters import sort_contexts
from src.retrieval.metadata_ranking import rerank_with_metadata


def search_indico_parent_contexts(
    child_collection: Any,
    embedder: Any,
    query: str,
    reranker: Any,
    *,
    candidate_count: int = 30,
    top_k: int = 5,
    person: str | None = None,
    title: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    sort_by: str = "relevance",
) -> dict[str, Any]:
    """Retrieve and reconstruct relevant Indico presentation contexts."""

    if not query.strip() or candidate_count <= 0 or top_k <= 0:
        return {"contexts": [], "references": []}

    filters = {
        "person": person,
        "title": title,
        "date_from": date_from,
        "date_to": date_to,
    }

    semantic = search_indico_children(
        child_collection,
        embedder,
        query,
        candidate_count=candidate_count,
        **filters,
    )

    keyword = search_indico_children_bm25(
        child_collection,
        query,
        candidate_count=candidate_count,
        **filters,
    )

    merged = merge_child_candidates(semantic, keyword)
    if not merged:
        return {"contexts": [], "references": []}

    ranked = rerank_child_candidates(query, merged, reranker)
    ranked = rerank_with_metadata(query, ranked, source="indico")

    contexts = []
    seen_parents = set()

    for child in ranked:
        parent_id = str(child["parent_id"])
        indico_id = str(child["indico_id"])
        # Treat all parent chunks from one attachment as one presentation.
        attachment_id = child.get("attachment_id") or child.get("attachment_url")
        identity = (indico_id, attachment_id or parent_id)

        if identity in seen_parents:
            continue

        parent = reconstruct_indico_parent(child_collection, child)
        seen_parents.add(identity)

        contexts.append({
            "source": "indico",
            "indico_id": indico_id,
            "parent_id": parent_id,
            "parent": parent,
            "best_child_id": child["document_id"],
            "best_child_text": child["cleaned_text"],
            "reranker_score": child["reranker_score"],
            "metadata_score": child["metadata_score"],
            "combined_score": child["combined_score"],
            "retrieval_methods": child.get(
                "retrieval_methods",
                [child.get("retrieval_method", "unknown")],
            ),
        })

        if len(contexts) >= top_k:
            break

    if sort_by == "relevance":
        contexts.sort(
            key=lambda item: item["combined_score"],
            reverse=True,
        )
    else:
        contexts = sort_contexts(
            contexts,
            source="indico",
            sort_by=sort_by,
        )

    references = [
        {
            "source": "indico",
            "indico_id": context["indico_id"],
            "presentation_title": context["parent"].get("presentation_title", ""),
            "meeting_name": context["parent"].get("meeting_name", ""),
            "speaker_name": context["parent"].get("speaker_name", ""),
            "start_date": context["parent"].get("start_date", ""),
            "event_url": context["parent"].get("event_url", ""),
            "attachment_url": context["parent"].get("attachment_url", ""),
        }
        for context in contexts
    ]

    return {
        "contexts": contexts,
        "references": references,
    }
