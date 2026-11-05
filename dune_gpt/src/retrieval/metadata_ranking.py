"""Optional metadata relevance signals for hybrid retrieval."""

from typing import Any

from src.retrieval.metadata_filters import normalize_text


def metadata_overlap(query: str, value: Any) -> float:
    """Return the fraction of meaningful metadata tokens found in the query."""
    query_tokens = set(normalize_text(query).split())
    value_tokens = set(normalize_text(value).split())

    if not query_tokens or not value_tokens:
        return 0.0

    stopwords = {
        "a", "an", "and", "at", "by", "for", "from", "in",
        "of", "on", "or", "the", "to", "with", "what",
        "which", "were", "was", "is", "are", "did",
        "dune", "meeting", "presentation", "presented",
    }

    meaningful = value_tokens - stopwords
    if not meaningful:
        return 0.0

    return len(meaningful & query_tokens) / len(meaningful)


def metadata_ranking_score(
    query: str,
    child: dict[str, Any],
    *,
    source: str,
) -> float:
    """Calculate a modest, source-aware metadata relevance bonus."""

    if source == "indico":
        title = child.get("presentation_title", "")
        meeting = child.get("meeting_name", "")
    elif source == "docdb":
        title = child.get("title") or child.get("document_title", "")
        meeting = ""
    else:
        raise ValueError(f"Unsupported source: {source}")

    title_overlap = metadata_overlap(query, title)
    meeting_overlap = metadata_overlap(query, meeting)

    return 2.0 * title_overlap + 0.25 * meeting_overlap


def rerank_with_metadata(
    query: str,
    ranked_children: list[dict[str, Any]],
    *,
    source: str,
) -> list[dict[str, Any]]:
    """Combine cross-encoder scores with optional metadata signals."""

    results = []

    for child in ranked_children:
        bonus = metadata_ranking_score(query, child, source=source)
        result = {
            **child,
            "metadata_score": bonus,
            "combined_score": float(child["reranker_score"]) + bonus,
        }
        results.append(result)

    return sorted(
        results,
        key=lambda item: item["combined_score"],
        reverse=True,
    )
