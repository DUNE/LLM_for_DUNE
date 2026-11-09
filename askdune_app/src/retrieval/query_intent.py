"""Interpret basic retrieval preferences from natural-language queries."""

import re
from typing import Any


NEWEST_PATTERNS = (
    r"\blatest\b",
    r"\bmost recent\b",
    r"\bnewest\b",
    r"\bup[- ]to[- ]date\b",
    r"\brecently published\b",
)

OLDEST_PATTERNS = (
    r"\bearliest\b",
    r"\boldest\b",
    r"\bfirst published\b",
    r"\bfirst recorded\b",
)


def interpret_query(query: str) -> dict[str, Any]:
    """Return retrieval preferences inferred from a query.

    Ambiguous or ordinary questions retain relevance-based ranking.
    """
    if not isinstance(query, str):
        raise TypeError("query must be a string")

    normalized = " ".join(query.casefold().split())

    newest = any(
        re.search(pattern, normalized)
        for pattern in NEWEST_PATTERNS
    )
    oldest = any(
        re.search(pattern, normalized)
        for pattern in OLDEST_PATTERNS
    )

    if newest and not oldest:
        sort_by = "newest"
    elif oldest and not newest:
        sort_by = "oldest"
    else:
        sort_by = "relevance"

    return {
        "sort_by": sort_by,
    }
