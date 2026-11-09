"""Build child-only DocDB records from existing parent chunks.

This module does not modify ChromaDB. Database writes are implemented
separately after record generation and reconstruction are validated.
"""

from typing import Any

from src.extractors.docdb_token_chunker import chunk_docdb_text


def build_child_records(
    parent: dict[str, Any],
    target_tokens: int = 480,
) -> list[dict[str, Any]]:
    """Convert one DocDB parent chunk into ordered child records."""
    if parent.get("source") != "docdb":
        raise ValueError("Expected a DocDB parent chunk")

    parent_id = str(parent.get("document_id", ""))
    if not parent_id:
        raise ValueError("Parent chunk is missing document_id")

    text = parent.get("cleaned_text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"Parent {parent_id}: missing cleaned_text")

    if "docdb_version" not in parent:
        raise ValueError(f"Parent {parent_id}: missing docdb_version")

    # Existing parent IDs use the form <docdb_id>_<chunk_number>.
    docdb_id, separator, chunk_number = parent_id.partition("_")
    if (
        not separator
        or not docdb_id.isdigit()
        or not chunk_number.isdigit()
    ):
        raise ValueError(f"Unexpected DocDB parent ID: {parent_id}")

    children = chunk_docdb_text(text, target_tokens=target_tokens)
    if not children:
        raise ValueError(f"Parent {parent_id}: no children generated")

    metadata = {
        key: value
        for key, value in parent.items()
        if key not in ("document_id", "cleaned_text", "raw_text")
    }

    records = []
    for index, child_text in enumerate(children):
        child = {
            **metadata,
            "document_id": f"docdb_child_{parent_id}_{index + 1}",
            "cleaned_text": child_text,
            "docdb_id": docdb_id,
            "parent_id": parent_id,
            "child_index": index,
            "child_count": len(children),
        }
        records.append(child)

    return records
