"""Compatibility adapter for the AskDUNE answer-generation pipeline."""

from typing import Any


def format_parent_child_results(
    result: dict[str, Any],
) -> tuple[list[str], list[str]]:
    """Convert unified retrieval results to (context snippets, URLs)."""

    snippets = []
    references = []
    seen_urls = set()

    for context in result.get("contexts", []):
        parent = context["parent"]
        source = context["source"]

        title = (
            parent.get("presentation_title")
            or parent.get("title")
            or parent.get("document_title")
            or "Untitled document"
        )

        text = (
            parent.get("cleaned_text")
            or parent.get("text")
            or parent.get("content")
            or ""
        )

        if not text:
            raise ValueError(
                f"Reconstructed {source} parent has no text: {title}"
            )

        snippets.append(
            f"Source: {source}\n"
            f"Title: {title}\n\n"
            f"{text}"
        )

    for reference in result.get("references", []):
        url = reference.get("url", "")

        if url and url not in seen_urls:
            references.append(url)
            seen_urls.add(url)

    return snippets, references
