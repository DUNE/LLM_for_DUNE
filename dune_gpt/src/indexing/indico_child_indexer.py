"""Build Indico child records from parent chunks.

No network access, embeddings, or database writes.
"""

from typing import Any

from src.extractors.docdb_token_chunker import chunk_docdb_text


def build_indico_child_records(
    parent: dict[str, Any],
    target_tokens: int = 480,
) -> list[dict[str, Any]]:
    """Convert one Indico parent into ordered token-limited children."""

    if parent.get("source") != "indico":
        raise ValueError("Expected an Indico parent")

    parent_id = str(parent.get("document_id", ""))
    if not parent_id:
        raise ValueError("Missing parent document_id")

    text = parent.get("cleaned_text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"Parent {parent_id}: missing cleaned_text")

    for field in ("indico_id", "event_url", "attachment_url"):
        if not parent.get(field):
            raise ValueError(f"Parent {parent_id}: missing {field}")

    child_texts = chunk_docdb_text(
        text,
        target_tokens=target_tokens,
    )

    if not child_texts:
        raise ValueError(f"Parent {parent_id}: no children generated")

    metadata = {
        key: value
        for key, value in parent.items()
        if key not in ("document_id", "cleaned_text", "raw_text")
    }

    records = []

    for index, child_text in enumerate(child_texts):
        records.append({
            **metadata,
            "document_id": f"indico_child_{parent_id}_{index + 1}",
            "cleaned_text": child_text,
            "parent_id": parent_id,
            "child_index": index,
            "child_count": len(child_texts),
        })

    return records
