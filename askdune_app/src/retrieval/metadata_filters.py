"""Metadata filtering and chronological ordering for DocDB and Indico."""

import re
import unicodedata
from datetime import date
from typing import Any


def normalize_text(value: Any) -> str:
    """Normalize case, accents, punctuation, and whitespace."""
    value = unicodedata.normalize("NFKD", str(value or ""))
    value = "".join(c for c in value if not unicodedata.combining(c))
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def matches_person(value: Any, person: str) -> bool:
    """Match individual names regardless of order, case, or accents."""
    requested = normalize_text(person).split()
    if not requested:
        return True

    if isinstance(value, (list, tuple)):
        return any(matches_person(entry, person) for entry in value)

    raw = str(value or "").strip()
    if not raw:
        return False

    parts = [part.strip() for part in raw.split(",") if part.strip()]

    # Indico typically stores names as "Last, First" pairs.
    # Check each pair separately to prevent cross-speaker matches.
    if len(parts) >= 2 and len(parts) % 2 == 0:
        candidates = [
            f"{parts[i]} {parts[i + 1]}"
            for i in range(0, len(parts), 2)
        ]
    else:
        candidates = [raw]

    for candidate in candidates:
        available = normalize_text(candidate).split()
        if all(token in available for token in requested):
            return True

    return False


def parse_date(value: Any) -> date | None:
    """Parse an ISO date or the date portion of an ISO timestamp."""
    if not value:
        return None
    try:
        return date.fromisoformat(str(value).strip()[:10])
    except ValueError:
        return None


def matches_metadata(
    metadata: dict[str, Any],
    *,
    source: str,
    person: str | None = None,
    title: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> bool:
    """Apply source-aware person, title, and inclusive date filters."""
    if source not in ("docdb", "indico"):
        raise ValueError(f"Unsupported source: {source}")

    if source == "indico" and metadata.get("index_status") != "complete":
        return False

    person_fields = (
        ("speaker_name", "conveners")
        if source == "indico"
        else ("authors", "author")
    )
    title_fields = (
        ("presentation_title", "meeting_name")
        if source == "indico"
        else ("title", "document_title")
    )
    date_fields = (
        ("start_date",)
        if source == "indico"
        else ("created_date", "content_last_modified_date", "date", "creation_date", "created_at", "document_date")
    )

    if person and not any(
        matches_person(metadata.get(field), person)
        for field in person_fields
    ):
        return False

    if title:
        requested = normalize_text(title)
        if not any(
            requested in normalize_text(metadata.get(field))
            for field in title_fields
        ):
            return False

    if date_from or date_to:
        dates = [parse_date(metadata.get(field)) for field in date_fields]
        document_date = next((d for d in dates if d is not None), None)
        if document_date is None:
            return False

        lower = parse_date(date_from) if date_from else None
        upper = parse_date(date_to) if date_to else None
        if date_from and lower is None:
            raise ValueError(f"Invalid date_from: {date_from}")
        if date_to and upper is None:
            raise ValueError(f"Invalid date_to: {date_to}")

        if lower and document_date < lower:
            return False
        if upper and document_date > upper:
            return False

    return True


def sort_contexts(
    contexts: list[dict[str, Any]],
    *,
    source: str,
    sort_by: str = "relevance",
) -> list[dict[str, Any]]:
    """Sort by relevance or date, including mixed DocDB/Indico results."""

    if source not in ("docdb", "indico", "unified"):
        raise ValueError(f"Unsupported source: {source}")

    if sort_by == "relevance":
        return sorted(
            contexts,
            key=lambda item: item.get(
                "combined_score",
                item.get("reranker_score", float("-inf")),
            ),
            reverse=True,
        )

    if sort_by not in ("newest", "oldest"):
        raise ValueError(f"Unsupported sort_by: {sort_by}")

    def date_key(item: dict[str, Any]) -> tuple[bool, int]:
        item_source = item.get("source") if source == "unified" else source

        if item_source == "indico":
            date_fields = ("start_date",)
        elif item_source == "docdb":
            date_fields = (
                "created_date",
                "content_last_modified_date",
                "date",
                "creation_date",
                "created_at",
                "document_date",
            )
        else:
            raise ValueError(f"Unsupported item source: {item_source}")

        metadata = item.get("parent", item)

        found = next(
            (
                parsed
                for field in date_fields
                if (parsed := parse_date(metadata.get(field))) is not None
            ),
            None,
        )

        if found is None:
            return (True, 0)

        ordinal = found.toordinal()
        return (False, -ordinal if sort_by == "newest" else ordinal)

    return sorted(contexts, key=date_key)
