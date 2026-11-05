"""Reconstruct DocDB parent contexts from ordered child records."""

from typing import Any


def reconstruct_parent(
    children: list[dict[str, Any]],
) -> dict[str, Any]:
    """Reconstruct one complete parent from its child records."""
    from src.indexing.docdb_child_validation import validate_parent_children

    ordered = validate_parent_children(children)

    parent_id = str(ordered[0]["parent_id"])

    metadata = {
        key: value
        for key, value in ordered[0].items()
        if key not in (
            "document_id",
            "cleaned_text",
            "raw_text",
            "docdb_id",
            "parent_id",
            "child_index",
            "child_count",
        )
    }

    return {
        **metadata,
        "document_id": parent_id,
        "cleaned_text": " ".join(
            child["cleaned_text"] for child in ordered
        ),
    }
