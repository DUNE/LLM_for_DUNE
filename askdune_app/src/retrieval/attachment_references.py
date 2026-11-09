"""Select distinct attachment references from ranked parent contexts."""

from typing import Any


def select_attachment_references(
    parent_contexts: list[dict[str, Any]],
    *,
    max_references: int = 3,
) -> list[dict[str, Any]]:
    """Return up to max_references unique source attachments."""

    if max_references <= 0:
        raise ValueError("max_references must be positive")

    references = []
    seen_urls = set()

    for context in parent_contexts:
        parent = context.get("parent", {})

        if not isinstance(parent, dict):
            continue

        url = str(parent.get("event_url") or "").strip()
        if not url:
            continue

        if url in seen_urls:
            continue

        seen_urls.add(url)

        references.append({
            "docdb_id": str(
                parent.get("docdb_id")
                or context.get("docdb_id", "")
            ),
            "parent_id": context.get("parent_id"),
            "filename": parent.get("filename", ""),
            "title": parent.get("title", ""),
            "document_type": parent.get("document_type", ""),
            "event_url": url,
            "reranker_score": context.get("reranker_score"),
        })

        if len(references) >= max_references:
            break

    return references
