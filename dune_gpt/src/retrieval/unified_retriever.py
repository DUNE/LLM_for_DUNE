"""Unified experimental retrieval across DocDB and Indico."""

from typing import Any
import re
from urllib.parse import urlsplit

from src.retrieval.query_intent import interpret_query
from src.retrieval.metadata_filters import sort_contexts
from src.retrieval.child_semantic_search import search_docdb_children
from src.retrieval.child_keyword_search import search_docdb_children_bm25
from src.retrieval.indico_semantic_search import search_indico_children
from src.retrieval.indico_keyword_search import search_indico_children_bm25
from src.retrieval.hybrid_reranker import (
    merge_child_candidates,
    rerank_child_candidates,
)
from src.retrieval.metadata_filters import matches_metadata
from src.retrieval.metadata_candidate_search import search_metadata_children
from src.retrieval.metadata_ranking import (
    metadata_ranking_score,
)
from src.retrieval.docdb_retriever import reconstruct_active_parent
from src.retrieval.indico_parent_reconstruction import reconstruct_indico_parent


def normalize_reference_url(value: Any) -> str:
    """Return a plain HTTP(S) URL from a stored URL or Markdown link."""
    raw = str(value or "").strip()

    # Markdown link: [description](https://example.org/file.pdf)
    if raw.startswith("[") and "](" in raw and raw.endswith(")"):
        raw = raw.split("](", 1)[1][:-1]

    # Undo Markdown escaping of URL query parameters.
    raw = raw.replace(chr(92) + "&", "&")

    parsed = urlsplit(raw)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return ""

    return raw


def search_unified_parent_contexts(
    docdb_collection: Any,
    manifest_collection: Any,
    indico_collection: Any,
    embedder: Any,
    query: str,
    reranker: Any,
    *,
    candidate_count: int = 30,
    top_k: int = 5,
    max_references: int = 5,
    person: str | None = None,
    title: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    sort_by: str | None = None,
) -> dict[str, Any]:
    """Search both sources, rerank jointly, and reconstruct selected parents."""

    if not query or not query.strip():
        raise ValueError("Search query cannot be empty")
    if sort_by is None:
        sort_by = interpret_query(query)["sort_by"]

    if candidate_count <= 0 or top_k <= 0 or max_references <= 0:
        raise ValueError("Retrieval limits must be positive")

    docdb_semantic = search_docdb_children(
        docdb_collection,
        manifest_collection,
        query,
        candidate_count=candidate_count,
        embedder=embedder,
    )
    docdb_keyword = search_docdb_children_bm25(
        docdb_collection,
        manifest_collection,
        query,
        candidate_count=candidate_count,
    )

    indico_semantic = search_indico_children(
        indico_collection,
        embedder,
        query,
        candidate_count=candidate_count,
    )
    indico_keyword = search_indico_children_bm25(
        indico_collection,
        query,
        candidate_count=candidate_count,
    )

    semantic = [
        {**child, "source": "docdb"} for child in docdb_semantic
    ] + [
        {**child, "source": "indico"} for child in indico_semantic
    ]
    keyword = [
        {**child, "source": "docdb"} for child in docdb_keyword
    ] + [
        {**child, "source": "indico"} for child in indico_keyword
    ]

    # Apply metadata constraints only when explicitly supplied.
    # Otherwise, all eligible physics content remains searchable.
    if any(value is not None for value in (
        person, title, date_from, date_to
    )):
        semantic = [
            child for child in semantic
            if matches_metadata(
                child,
                source=child["source"],
                person=person,
                title=title,
                date_from=date_from,
                date_to=date_to,
            )
        ]
        keyword = [
            child for child in keyword
            if matches_metadata(
                child,
                source=child["source"],
                person=person,
                title=title,
                date_from=date_from,
                date_to=date_to,
            )
        ]

    # Direct metadata lookup ensures matching records are not missed
    # merely because they fall outside semantic/BM25 candidate limits.
    metadata_candidates = []

    if any(value is not None for value in (
        person, title, date_from, date_to
    )):
        metadata_candidates.extend(search_metadata_children(
            docdb_collection,
            source="docdb",
            manifest_collection=manifest_collection,
            person=person,
            title=title,
            date_from=date_from,
            date_to=date_to,
        ))

        metadata_candidates.extend(search_metadata_children(
            indico_collection,
            source="indico",
            person=person,
            title=title,
            date_from=date_from,
            date_to=date_to,
        ))

    merged = merge_child_candidates(
        semantic,
        keyword + metadata_candidates,
    )
    ranked = rerank_child_candidates(query, merged, reranker)

    for child in ranked:
        bonus = metadata_ranking_score(
            query,
            child,
            source=child["source"],
        )
        child["metadata_score"] = bonus
        child["combined_score"] = child["reranker_score"] + bonus

    ranked = sort_contexts(ranked, source="unified", sort_by=sort_by)

    contexts = []
    references = []
    seen_parents = set()
    seen_references = set()

    for child in ranked:
        source = child["source"]

        if source == "docdb":
            identity = (
                source,
                str(child["docdb_id"]),
                str(child["parent_id"]),
            )
        else:
            identity = (
                source,
                str(child["indico_id"]),
                str(child.get("attachment_id")
                    or child.get("attachment_url")
                    or child["parent_id"]),
            )

        if identity in seen_parents:
            continue

        if source == "docdb":
            parent = reconstruct_active_parent(
                docdb_collection,
                manifest_collection,
                child,
            )
        else:
            parent = reconstruct_indico_parent(
                indico_collection,
                child,
            )

        seen_parents.add(identity)

        context = {
            "source": source,
            "parent_id": child["parent_id"],
            "parent": parent,
            "best_child_id": child["document_id"],
            "best_child_text": child["cleaned_text"],
            "reranker_score": child["reranker_score"],
            "metadata_score": child["metadata_score"],
            "combined_score": child["combined_score"],
            "retrieval_methods": child.get("retrieval_methods", []),
        }
        contexts.append(context)

        url = (
            parent.get("attachment_url")
            or parent.get("docdb_url")
            or parent.get("event_url")
        )

        url = normalize_reference_url(url)

        if url and url not in seen_references:
            seen_references.add(url)
            if len(references) < max_references:
                references.append({
                    "source": source,
                    "url": url,
                    "title": (
                        parent.get("presentation_title")
                        or parent.get("title")
                        or parent.get("document_title")
                        or ""
                    ),
                })

        if len(contexts) >= top_k:
            break

    return {
        "contexts": contexts,
        "references": references,
    }
