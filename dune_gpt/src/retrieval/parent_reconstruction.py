"""Reconstruct DocDB or Indico parent context from child records."""

from typing import Any

from src.retrieval.parent_child_validation import validate_children


def reconstruct_parent_context(
    children: list[dict[str, Any]],
    *,
    source: str,
) -> dict[str, Any]:
    """Validate and reconstruct one parent from its ordered children."""

    ordered = validate_children(children, source=source)
    parent_id = str(ordered[0]["parent_id"])

    excluded = {
        "document_id",
        "cleaned_text",
        "raw_text",
        "parent_id",
        "child_index",
        "child_count",
        "index_status",
        "vector_id",
    }

    metadata = {
        key: value
        for key, value in ordered[0].items()
        if key not in excluded
    }

    return {
        **metadata,
        "document_id": parent_id,
        "cleaned_text": " ".join(
            child["cleaned_text"] for child in ordered
        ),
    }
