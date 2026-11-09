"""Shared validation for DocDB parent-child records."""

from typing import Any


def validate_parent_children(
    children: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Validate a complete parent and return ordered children."""
    if not children:
        raise ValueError("Cannot reconstruct parent from empty children")

    parent_id = str(children[0].get("parent_id", ""))
    docdb_id = str(children[0].get("docdb_id", ""))
    version = str(children[0].get("docdb_version", ""))
    expected_count = children[0].get("child_count")

    if not parent_id or not docdb_id or not version:
        raise ValueError("Missing parent identity or DocDB version")

    if type(expected_count) is not int or expected_count <= 0:
        raise ValueError("Invalid child_count")

    indices = set()

    for child in children:
        if str(child.get("parent_id", "")) != parent_id:
            raise ValueError("Children belong to different parents")

        if str(child.get("docdb_id", "")) != docdb_id:
            raise ValueError("Inconsistent DocDB document identity")

        if str(child.get("docdb_version", "")) != version:
            raise ValueError("Mixed DocDB versions")

        if child.get("child_count") != expected_count:
            raise ValueError("Inconsistent child_count")

        index = child.get("child_index")
        if type(index) is not int or index < 0:
            raise ValueError("Invalid child_index")

        if index in indices:
            raise ValueError("Duplicate child_index")

        if not str(child.get("cleaned_text", "")).strip():
            raise ValueError("Empty child text")

        indices.add(index)

    if len(children) != expected_count:
        raise ValueError(
            f"Parent {parent_id}: expected {expected_count} children, "
            f"received {len(children)}"
        )

    if indices != set(range(expected_count)):
        raise ValueError(f"Parent {parent_id}: missing child indices")

    # All children of a parent must inherit identical document metadata.
    # Exclude fields that legitimately differ between child chunks.
    child_specific_fields = {
        "document_id",
        "cleaned_text",
        "raw_text",
        "child_index",
        "vector_id",
    }

    reference = children[0]
    metadata_fields = set().union(*(child.keys() for child in children))
    metadata_fields -= child_specific_fields

    for child in children[1:]:
        for field in metadata_fields:
            if child.get(field) != reference.get(field):
                raise ValueError(
                    f"Parent {parent_id}: inconsistent metadata field {field}"
                )

    ordered = sorted(children, key=lambda child: child["child_index"])

    return ordered
