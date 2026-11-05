"""Build parent chunks from extracted Indico attachments.

Each attachment is processed independently. No network or database access.
"""

import hashlib
from typing import Any


def build_indico_parents(
    attachment: dict[str, Any],
    chunk_size: int = 2000,
) -> list[dict[str, Any]]:
    """Build ordered parent chunks from one Indico attachment."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    event_id = attachment.get("indico_id")
    if event_id is None or not str(event_id).strip():
        raise ValueError("Missing indico_id")

    event_url = attachment.get("event_url")
    attachment_url = attachment.get("attachment_url")

    if not isinstance(event_url, str) or not event_url.strip():
        raise ValueError("Missing Indico event_url")

    if not isinstance(attachment_url, str) or not attachment_url.strip():
        raise ValueError("Missing attachment_url")

    raw_text = attachment.get("cleaned_text")
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise ValueError("Missing extracted text")

    words = raw_text.split()
    attachment_hash = hashlib.sha256(
        attachment_url.encode("utf-8")
    ).hexdigest()[:16]

    metadata = {
        key: value
        for key, value in attachment.items()
        if key not in ("document_id", "cleaned_text", "raw_text")
    }

    metadata.update({
        "source": "indico",
        "indico_id": str(event_id),
        "attachment_id": f"{event_id}_{attachment_hash}",
    })

    total_parents = (len(words) + chunk_size - 1) // chunk_size
    parents = []

    for index in range(total_parents):
        start = index * chunk_size
        chunk_words = words[start:start + chunk_size]

        parents.append({
            **metadata,
            "document_id": (
                f"indico_{event_id}_{attachment_hash}_{index + 1}"
            ),
            "cleaned_text": " ".join(chunk_words),
            "parent_index": index,
            "parent_count": total_parents,
        })

    return parents
