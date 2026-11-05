"""Source-independent validation of ordered parent-child records."""

from typing import Any


def validate_children(
    children: list[dict[str, Any]],
    *,
    source: str,
) -> list[dict[str, Any]]:
    """Validate a complete parent and return children in index order."""

    if source not in ("docdb", "indico"):
        raise ValueError(f"Unsupported source: {source}")

    if not children:
        raise ValueError("Cannot reconstruct parent from empty children")

    first = children[0]
    parent_id = str(first.get("parent_id") or "")
    identity_field = "docdb_id" if source == "docdb" else "indico_id"
    identity = str(first.get(identity_field) or "")
    expected_count = first.get("child_count")

    if not parent_id or not identity:
        raise ValueError("Missing parent or source identity")

    if type(expected_count) is not int or expected_count <= 0:
        raise ValueError("Invalid child_count")

    version = str(first.get("docdb_version") or "") if source == "docdb" else None
    if source == "docdb" and not version:
        raise ValueError("Missing DocDB version")

    indices = set()

    for child in children:
        if str(child.get("parent_id") or "") != parent_id:
            raise ValueError("Children belong to different parents")

        if str(child.get(identity_field) or "") != identity:
            raise ValueError("Inconsistent source identity")

        if source == "docdb" and str(child.get("docdb_version") or "") != version:
            raise ValueError("Mixed DocDB versions")

        if source == "indico" and child.get("index_status") != "complete":
            raise ValueError("Incomplete Indico child")

        if child.get("child_count") != expected_count:
            raise ValueError("Inconsistent child_count")

        index = child.get("child_index")
        if type(index) is not int or index < 0 or index in indices:
            raise ValueError("Invalid or duplicate child_index")

        if not str(child.get("cleaned_text") or "").strip():
            raise ValueError("Empty child text")

        indices.add(index)

    if len(children) != expected_count or indices != set(range(expected_count)):
        raise ValueError(f"Parent {parent_id}: missing or extra children")

    excluded = {
        "document_id", "cleaned_text", "raw_text",
        "child_index", "vector_id",
    }
    fields = set().union(*(child.keys() for child in children)) - excluded

    for child in children[1:]:
        for field in fields:
            if child.get(field) != first.get(field):
                raise ValueError(
                    f"Parent {parent_id}: inconsistent metadata field {field}"
                )

    return sorted(children, key=lambda child: child["child_index"])
