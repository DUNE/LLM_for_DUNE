"""Version-aware storage helpers for experimental DocDB child indexing.

Parents are reconstructed from children and are never stored separately.
This module is experimental and must not modify production collections.
"""

from typing import Any


def versioned_child_id(child: dict[str, Any]) -> str:
    """Return an ID that uniquely identifies a child within a DocDB version."""
    docdb_id = str(child.get("docdb_id", ""))
    version = str(child.get("docdb_version", ""))
    parent_id = str(child.get("parent_id", ""))
    child_index = child.get("child_index")

    if not docdb_id.isdigit():
        raise ValueError("Invalid DocDB document ID")
    if not version.isdigit():
        raise ValueError("Invalid DocDB version")
    if not parent_id.startswith(f"{docdb_id}_"):
        raise ValueError("Parent does not belong to DocDB document")
    if type(child_index) is not int or child_index < 0:
        raise ValueError("Invalid child index")

    return (
        f"docdb_child_{docdb_id}_v{version}_"
        f"{parent_id}_{child_index + 1}"
    )


def active_version_key(docdb_id: str | int) -> str:
    """Return the persistent manifest key for a DocDB document."""
    docdb_id = str(docdb_id)

    if not docdb_id.isdigit():
        raise ValueError("Invalid DocDB document ID")

    return f"docdb_active_version_{docdb_id}"


def validate_newer_version(
    incoming_version: str | int,
    active_version: str | int | None,
) -> bool:
    """Return True only when incoming version is newer than active version."""
    incoming = int(incoming_version)

    if incoming < 1:
        raise ValueError("DocDB version must be positive")

    if active_version is None:
        return True

    return incoming > int(active_version)


def get_active_version(manifest_collection: Any, docdb_id: str | int) -> int | None:
    """Read the persistent active version for a DocDB document."""
    key = active_version_key(docdb_id)
    result = manifest_collection.get(ids=[key], include=["metadatas"])

    if not result["ids"]:
        return None

    metadata = result["metadatas"][0]
    if not metadata or "active_version" not in metadata:
        raise ValueError(f"Invalid active-version manifest: {key}")

    version = int(metadata["active_version"])
    if version < 1:
        raise ValueError(f"Invalid active version for DocDB {docdb_id}")

    return version


def set_active_version(
    manifest_collection: Any,
    docdb_id: str | int,
    version: str | int,
) -> None:
    """Persist the active version after the new child set is verified."""
    version = int(version)
    if version < 1:
        raise ValueError("DocDB version must be positive")

    key = active_version_key(docdb_id)

    manifest_collection.upsert(
        ids=[key],
        documents=[f"Active DocDB version {version}"],
        metadatas=[{
            "docdb_id": str(docdb_id),
            "active_version": version,
        }],
    )

    stored = get_active_version(manifest_collection, docdb_id)
    if stored != version:
        raise RuntimeError(
            f"Failed to activate DocDB {docdb_id} version {version}"
        )


def get_all_active_versions(manifest_collection: Any) -> dict[str, int]:
    """Return active DocDB versions without loading child text or embeddings."""
    result = manifest_collection.get(include=["metadatas"])

    versions = {}

    for record_id, metadata in zip(
        result["ids"], result["metadatas"]
    ):
        # Staging records are not active-version records.
        if record_id.startswith("docdb_staged_"):
            continue

        if not metadata:
            raise ValueError(f"Missing manifest metadata: {record_id}")

        docdb_id = str(metadata.get("docdb_id", ""))
        expected_id = active_version_key(docdb_id)

        if record_id != expected_id:
            raise ValueError(f"Invalid manifest record ID: {record_id}")

        version = int(metadata["active_version"])

        if version < 1:
            raise ValueError(f"Invalid active version for DocDB {docdb_id}")

        if docdb_id in versions:
            raise ValueError(f"Duplicate manifest entry for DocDB {docdb_id}")

        versions[docdb_id] = version

    return versions


def get_active_child_backup(
    child_collection: Any,
    manifest_collection: Any,
    docdb_id: str | int,
) -> dict[str, Any]:
    """Back up active child text, metadata, and embeddings for one document."""
    docdb_id = str(docdb_id)
    active_version = get_active_version(manifest_collection, docdb_id)

    if active_version is None:
        return {
            "version": None,
            "ids": [],
            "documents": [],
            "metadatas": [],
            "embeddings": [],
        }

    result = child_collection.get(
        where={
            "$and": [
                {"docdb_id": docdb_id},
                {"docdb_version": str(active_version)},
            ]
        },
        include=["documents", "metadatas", "embeddings"],
    )

    if not result["ids"]:
        raise ValueError(
            f"DocDB {docdb_id}: active version {active_version} "
            "has no indexed children"
        )

    for metadata in result["metadatas"]:
        if (
            str(metadata["docdb_id"]) != docdb_id
            or int(metadata["docdb_version"]) != active_version
        ):
            raise ValueError("Backup contains inconsistent DocDB metadata")

    return {
        "version": active_version,
        "ids": result["ids"],
        "documents": result["documents"],
        "metadatas": result["metadatas"],
        "embeddings": result["embeddings"],
    }


def validate_child_backup(backup: dict[str, Any]) -> int:
    """Validate all backed-up children and return the parent count."""
    from collections import defaultdict
    from src.indexing.docdb_child_validation import validate_parent_children

    version = backup.get("version")
    ids = backup.get("ids", [])
    documents = backup.get("documents", [])
    metadatas = backup.get("metadatas", [])
    embeddings = backup.get("embeddings")

    if version is None:
        if ids:
            raise ValueError("New-document backup unexpectedly contains children")
        return 0

    if not ids or len(ids) != len(documents) or len(ids) != len(metadatas):
        raise ValueError("Incomplete child backup")

    if embeddings is None or len(embeddings) != len(ids):
        raise ValueError("Missing or incomplete backup embeddings")

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate child IDs in backup")

    parents = defaultdict(list)
    docdb_ids = set()

    for record_id, text, metadata in zip(ids, documents, metadatas):
        if int(metadata["docdb_version"]) != int(version):
            raise ValueError("Mixed versions in backup")

        docdb_ids.add(str(metadata["docdb_id"]))

        parents[metadata["parent_id"]].append({
            **metadata,
            "document_id": record_id,
            "cleaned_text": text,
        })

    if len(docdb_ids) != 1:
        raise ValueError("Backup contains multiple DocDB documents")

    for records in parents.values():
        validate_parent_children(records)

    return len(parents)


def validate_staged_children(
    children: list[dict[str, Any]],
    docdb_id: str | int,
    version: str | int,
) -> int:
    """Validate a complete new child set before writing it to ChromaDB."""
    from collections import defaultdict
    from src.indexing.docdb_child_validation import validate_parent_children

    docdb_id = str(docdb_id)
    version = int(version)

    if version < 1 or not children:
        raise ValueError("Invalid version or empty staged child set")

    record_ids = set()
    parents = defaultdict(list)

    for child in children:
        if str(child.get("docdb_id")) != docdb_id:
            raise ValueError("Staged child belongs to another DocDB document")

        if int(child.get("docdb_version", 0)) != version:
            raise ValueError("Staged child has an incorrect version")

        record_id = versioned_child_id(child)

        if record_id in record_ids:
            raise ValueError(f"Duplicate staged child ID: {record_id}")

        record_ids.add(record_id)

        if not str(child.get("cleaned_text", "")).strip():
            raise ValueError(f"Empty staged child: {record_id}")

        parents[child["parent_id"]].append(child)

    for parent_id, siblings in parents.items():
        validated = validate_parent_children(siblings)

        if validated[0]["parent_id"] != parent_id:
            raise ValueError(f"Incorrect reconstructed parent: {parent_id}")

        if int(validated[0]["docdb_version"]) != version:
            raise ValueError(f"Incorrect parent version: {parent_id}")

    return len(parents)


def stage_docdb_children(
    child_collection: Any,
    manifest_collection: Any,
    docdb_id: str | int,
    version: str | int,
    children: list[dict[str, Any]],
    embeddings: Any,
) -> dict[str, int]:
    """Stage and verify a newer DocDB version without activating it."""
    import numpy as np

    docdb_id = str(docdb_id)
    version = int(version)

    active_version = get_active_version(manifest_collection, docdb_id)

    if not validate_newer_version(version, active_version):
        raise ValueError(
            f"DocDB {docdb_id}: version {version} is not newer "
            f"than active version {active_version}"
        )

    parent_count = validate_staged_children(children, docdb_id, version)

    vectors = np.asarray(embeddings, dtype=np.float32)

    if (
        vectors.ndim != 2
        or vectors.shape[0] != len(children)
        or vectors.shape[1] == 0
        or not np.isfinite(vectors).all()
    ):
        raise ValueError("Invalid or incomplete staged embeddings")

    ids = [versioned_child_id(child) for child in children]
    documents = [child["cleaned_text"] for child in children]

    metadatas = [
        {
            key: value
            for key, value in child.items()
            if key not in ("cleaned_text", "document_id")
            and value is not None
        }
        for child in children
    ]

    # Validate all metadata before starting a database write.
    for metadata in metadatas:
        for key, value in metadata.items():
            if (
                not isinstance(key, str)
                or not key
                or not isinstance(value, (str, int, float, bool))
            ):
                raise ValueError(
                    f"Unsupported ChromaDB metadata: {key!r}={value!r}"
                )

    # Reject unexpected existing records rather than overwrite them.
    existing = child_collection.get(ids=ids, include=[])
    if existing["ids"]:
        raise ValueError(
            f"Staged IDs already exist for DocDB {docdb_id} "
            f"version {version}; recovery or cleanup is required"
        )

    # Stage children in bounded batches to avoid duplicating
    # the entire embedding array as Python lists.
    batch_size = 32

    for offset in range(0, len(ids), batch_size):
        stop = offset + batch_size

        child_collection.add(
            ids=ids[offset:stop],
            documents=documents[offset:stop],
            metadatas=metadatas[offset:stop],
            embeddings=vectors[offset:stop].tolist(),
        )

    stored = child_collection.get(
        where={
            "$and": [
                {"docdb_id": docdb_id},
                {"docdb_version": str(version)},
            ]
        },
        include=["documents", "metadatas"],
    )

    if set(stored["ids"]) != set(ids):
        raise RuntimeError("Staged child IDs do not match expected IDs")

    from collections import defaultdict
    from src.indexing.docdb_child_validation import validate_parent_children

    parents = defaultdict(list)

    for record_id, text, metadata in zip(
        stored["ids"], stored["documents"], stored["metadatas"]
    ):
        parents[metadata["parent_id"]].append({
            **metadata,
            "document_id": record_id,
            "cleaned_text": text,
        })

    for siblings in parents.values():
        validate_parent_children(siblings)

    if len(parents) != parent_count:
        raise RuntimeError("Staged parent count does not match")

    if get_active_version(manifest_collection, docdb_id) != active_version:
        raise RuntimeError("Active version changed during staging")

    return {
        "children": len(ids),
        "parents": parent_count,
        "version": version,
    }


def activate_staged_docdb_version(
    child_collection: Any,
    manifest_collection: Any,
    docdb_id: str | int,
    version: str | int,
) -> dict[str, int]:
    """Activate a fully staged DocDB version, retaining old children."""
    import numpy as np
    from collections import defaultdict
    from src.indexing.docdb_child_validation import validate_parent_children

    docdb_id = str(docdb_id)
    version = int(version)

    previous_version = get_active_version(manifest_collection, docdb_id)

    if not validate_newer_version(version, previous_version):
        raise ValueError(
            f"DocDB {docdb_id}: cannot activate version {version}; "
            f"current active version is {previous_version}"
        )

    staged = child_collection.get(
        where={
            "$and": [
                {"docdb_id": docdb_id},
                {"docdb_version": str(version)},
            ]
        },
        include=["documents", "metadatas"],
    )

    if not staged["ids"]:
        raise ValueError("No staged children found")

    completion = manifest_collection.get(
        ids=[staging_record_key(docdb_id, version)],
        include=["metadatas"],
    )

    if len(completion["ids"]) != 1:
        raise ValueError("Staging-completion record is missing")

    record = completion["metadatas"][0]

    if not record or (
        record.get("status") != "complete"
        or str(record.get("docdb_id")) != docdb_id
        or int(record.get("staged_version", 0)) != version
        or int(record.get("previous_version", -1))
           != (previous_version or 0)
        or int(record.get("expected_children", 0)) != len(staged["ids"])
    ):
        raise ValueError("Staging-completion record is inconsistent")

    expected_fingerprint = record.get("child_id_fingerprint")

    if (
        not isinstance(expected_fingerprint, str)
        or len(expected_fingerprint) != 64
        or staged_child_fingerprint(staged["ids"]) != expected_fingerprint
    ):
        raise ValueError("Staged child-ID fingerprint mismatch")

    parents = defaultdict(list)

    for record_id, text, metadata in zip(
        staged["ids"], staged["documents"], staged["metadatas"]
    ):
        if record_id != versioned_child_id(metadata):
            raise ValueError(f"Invalid staged child ID: {record_id}")

        parents[metadata["parent_id"]].append({
            **metadata,
            "document_id": record_id,
            "cleaned_text": text,
        })

    for siblings in parents.values():
        validate_parent_children(siblings)

    if len(parents) != int(record["expected_parents"]):
        raise ValueError("Staged parent count does not match completion record")

    # Validate stored embeddings in bounded batches to limit RAM usage.
    embedding_dimension = None
    batch_size = 32

    for offset in range(0, len(staged["ids"]), batch_size):
        batch_ids = staged["ids"][offset:offset + batch_size]

        batch = child_collection.get(
            ids=batch_ids,
            include=["embeddings"],
        )

        vectors = batch["embeddings"]

        if vectors is None or len(vectors) != len(batch_ids):
            raise ValueError("Staged embeddings are incomplete")

        vectors = np.asarray(vectors, dtype=np.float32)

        if (
            vectors.ndim != 2
            or vectors.shape[0] != len(batch_ids)
            or vectors.shape[1] == 0
            or not np.isfinite(vectors).all()
        ):
            raise ValueError("Invalid staged embeddings")

        if embedding_dimension is None:
            embedding_dimension = vectors.shape[1]
        elif vectors.shape[1] != embedding_dimension:
            raise ValueError("Inconsistent staged embedding dimensions")

        del vectors, batch

    # Recheck immediately before activation.
    if get_active_version(manifest_collection, docdb_id) != previous_version:
        raise RuntimeError("Active version changed before activation")

    set_active_version(manifest_collection, docdb_id, version)

    if get_active_version(manifest_collection, docdb_id) != version:
        raise RuntimeError("Active-version activation verification failed")

    return {
        "previous_version": previous_version,
        "active_version": version,
        "children": len(staged["ids"]),
        "parents": len(parents),
    }


def staging_record_key(docdb_id: str | int, version: str | int) -> str:
    """Unique manifest key for a staged DocDB version."""
    docdb_id = str(docdb_id)
    version = int(version)

    active_version_key(docdb_id)

    if version < 1:
        raise ValueError("Staged version must be positive")

    return f"docdb_staged_{docdb_id}_v{version}"


def record_completed_staging(
    manifest_collection: Any,
    docdb_id: str | int,
    version: str | int,
    expected_children: int,
    expected_parents: int,
    previous_version: int | None,
    child_ids: list[str],
) -> None:
    """Record completion only after all staged children are verified."""
    docdb_id = str(docdb_id)
    version = int(version)

    if expected_children < 1 or expected_parents < 1:
        raise ValueError("Staged document must contain children and parents")

    if not validate_newer_version(version, previous_version):
        raise ValueError("Staged version must be newer than previous version")

    if get_active_version(manifest_collection, docdb_id) != previous_version:
        raise RuntimeError("Active version changed during staging")

    if len(child_ids) != expected_children:
        raise ValueError("Staging child-ID count does not match")

    if any(
        not record_id.startswith(f"docdb_child_{docdb_id}_v{version}_")
        for record_id in child_ids
    ):
        raise ValueError("Staging contains IDs from another document or version")

    fingerprint = staged_child_fingerprint(child_ids)

    key = staging_record_key(docdb_id, version)

    metadata = {
        "docdb_id": docdb_id,
        "staged_version": version,
        "expected_children": expected_children,
        "expected_parents": expected_parents,
        "child_id_fingerprint": fingerprint,
        "previous_version": previous_version or 0,
        "status": "complete",
    }

    manifest_collection.upsert(
        ids=[key],
        documents=[f"Completed staging for DocDB {docdb_id} v{version}"],
        metadatas=[metadata],
    )

    stored = manifest_collection.get(
        ids=[key],
        include=["metadatas"],
    )

    if (
        len(stored["ids"]) != 1
        or stored["metadatas"][0] != metadata
    ):
        raise RuntimeError("Staging-completion record verification failed")


def staged_child_fingerprint(child_ids: list[str]) -> str:
    """Compute a deterministic fingerprint of a complete child-ID set."""
    import hashlib
    import json

    if not child_ids:
        raise ValueError("Cannot fingerprint an empty child set")

    if len(child_ids) != len(set(child_ids)):
        raise ValueError("Duplicate child IDs in staging fingerprint")

    if any(not isinstance(record_id, str) or not record_id for record_id in child_ids):
        raise ValueError("Invalid child ID in staging fingerprint")

    payload = json.dumps(
        sorted(child_ids),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(payload).hexdigest()




def cleanup_activated_docdb_version(
    child_collection: Any,
    manifest_collection: Any,
    docdb_id: str,
    version: int,
    batch_size: int = 100,
) -> dict[str, int]:
    """Remove obsolete children and staging metadata after activation."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    docdb_id = str(docdb_id)
    version = int(version)

    if get_active_version(manifest_collection, docdb_id) != version:
        raise ValueError("Cannot clean up a version that is not active")

    stage_key = staging_record_key(docdb_id, version)
    stage = manifest_collection.get(
        ids=[stage_key],
        include=["metadatas"],
    )

    if len(stage["ids"]) != 1:
        raise ValueError("Completed staging record is missing")

    record = stage["metadatas"][0]

    if (
        record.get("status") != "complete"
        or str(record.get("docdb_id")) != docdb_id
        or int(record.get("staged_version", -1)) != version
    ):
        raise ValueError("Invalid staging-completion record")

    # Verify the active child set before removing the previous version.
    active = child_collection.get(
        where={
            "$and": [
                {"docdb_id": docdb_id},
                {"docdb_version": str(version)},
            ]
        },
        include=[],
    )

    if (
        len(active["ids"]) != int(record["expected_children"])
        or staged_child_fingerprint(active["ids"])
        != record.get("child_id_fingerprint")
    ):
        raise ValueError("Active child set failed verification")

    deleted = 0

    # Fetch only IDs and metadata, never text or embeddings.
    records = child_collection.get(
        where={"docdb_id": docdb_id},
        include=["metadatas"],
    )

    obsolete_ids = [
        record_id
        for record_id, metadata in zip(
            records["ids"], records["metadatas"]
        )
        if str(metadata.get("docdb_version")) != str(version)
    ]

    for offset in range(0, len(obsolete_ids), batch_size):
        if get_active_version(manifest_collection, docdb_id) != version:
            raise ValueError("Active version changed during cleanup")

        batch = obsolete_ids[offset:offset + batch_size]
        child_collection.delete(ids=batch)
        deleted += len(batch)

    # Confirm the active version remains intact.
    remaining = child_collection.get(
        where={"docdb_id": docdb_id},
        include=["metadatas"],
    )

    if (
        len(remaining["ids"]) != int(record["expected_children"])
        or any(
            str(metadata.get("docdb_version")) != str(version)
            for metadata in remaining["metadatas"]
        )
        or staged_child_fingerprint(remaining["ids"])
        != record["child_id_fingerprint"]
        or get_active_version(manifest_collection, docdb_id) != version
    ):
        raise ValueError("Post-cleanup verification failed")

    manifest_collection.delete(ids=[stage_key])

    return {
        "active_version": version,
        "active_children": len(remaining["ids"]),
        "obsolete_children_deleted": deleted,
    }


def stage_docdb_child_batch(
    child_collection: Any,
    manifest_collection: Any,
    docdb_id: str | int,
    version: str | int,
    children: list[dict[str, Any]],
    embeddings: Any,
) -> list[str]:
    """Persist one embedding batch without activating the document."""
    import numpy as np

    docdb_id = str(docdb_id)
    version = int(version)

    active_version = get_active_version(manifest_collection, docdb_id)

    if not validate_newer_version(version, active_version):
        raise ValueError(
            f"DocDB {docdb_id}: version {version} is not newer "
            f"than active version {active_version}"
        )

    if not children:
        raise ValueError("Cannot stage an empty child batch")

    vectors = np.asarray(embeddings, dtype=np.float32)

    if (
        vectors.ndim != 2
        or vectors.shape[0] != len(children)
        or vectors.shape[1] == 0
        or not np.isfinite(vectors).all()
    ):
        raise ValueError("Invalid embedding batch")

    ids = []
    documents = []
    metadatas = []

    for child in children:
        if str(child.get("docdb_id")) != docdb_id:
            raise ValueError("Child belongs to another DocDB document")

        if int(child.get("docdb_version", 0)) != version:
            raise ValueError("Child version mismatch")

        if not str(child.get("cleaned_text", "")).strip():
            raise ValueError("Child has empty text")

        child_id = versioned_child_id(child)
        ids.append(child_id)
        documents.append(child["cleaned_text"])

        metadata = {
            key: value
            for key, value in child.items()
            if key not in ("cleaned_text", "document_id")
            and value is not None
        }

        for key, value in metadata.items():
            if (
                not isinstance(key, str)
                or not key
                or not isinstance(value, (str, int, float, bool))
            ):
                raise ValueError(
                    f"Unsupported ChromaDB metadata: {key!r}={value!r}"
                )

        metadatas.append(metadata)

    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate child IDs within staging batch")

    existing = child_collection.get(ids=ids, include=[])

    if existing["ids"]:
        raise ValueError(
            f"Staged child IDs already exist: {existing['ids'][:3]}"
        )

    if get_active_version(manifest_collection, docdb_id) != active_version:
        raise RuntimeError("Active version changed before batch write")

    child_collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=vectors.tolist(),
    )

    stored = child_collection.get(
        ids=ids,
        include=["documents", "metadatas"],
    )

    if set(stored["ids"]) != set(ids):
        raise RuntimeError("Incomplete child batch write")

    stored_by_id = {
        record_id: (text, metadata)
        for record_id, text, metadata in zip(
            stored["ids"],
            stored["documents"],
            stored["metadatas"],
        )
    }

    for child_id, text, metadata in zip(ids, documents, metadatas):
        stored_text, stored_metadata = stored_by_id[child_id]

        if stored_text != text or stored_metadata != metadata:
            raise RuntimeError(
                f"Staged child verification failed: {child_id}"
            )

    if get_active_version(manifest_collection, docdb_id) != active_version:
        raise RuntimeError("Active version changed during batch staging")

    return ids


def record_verified_docdb_staging(
    child_collection: Any,
    manifest_collection: Any,
    docdb_id: str | int,
    version: str | int,
    expected_children: int,
    expected_parents: int,
    previous_version: int | None,
) -> None:
    """Verify a complete staged document before recording completion."""
    from collections import defaultdict
    from src.indexing.docdb_child_validation import validate_parent_children

    docdb_id = str(docdb_id)
    version = int(version)

    if expected_children < 1 or expected_parents < 1:
        raise ValueError("Expected child and parent counts must be positive")

    if not validate_newer_version(version, previous_version):
        raise ValueError("Staged version must be newer than previous version")

    if get_active_version(manifest_collection, docdb_id) != previous_version:
        raise RuntimeError("Active version changed during staging")

    stored = child_collection.get(
        where={
            "$and": [
                {"docdb_id": docdb_id},
                {"docdb_version": str(version)},
            ]
        },
        include=["documents", "metadatas"],
    )

    ids = stored["ids"]

    if len(ids) != expected_children:
        raise ValueError(
            f"Expected {expected_children} staged children, "
            f"found {len(ids)}"
        )

    parents = defaultdict(list)

    for record_id, text, metadata in zip(
        stored["ids"],
        stored["documents"],
        stored["metadatas"],
    ):
        if record_id != versioned_child_id(metadata):
            raise ValueError(f"Invalid staged child ID: {record_id}")

        if (
            str(metadata.get("docdb_id")) != docdb_id
            or int(metadata.get("docdb_version", 0)) != version
        ):
            raise ValueError("Staged child document/version mismatch")

        parents[metadata["parent_id"]].append({
            **metadata,
            "document_id": record_id,
            "cleaned_text": text,
        })

    if len(parents) != expected_parents:
        raise ValueError(
            f"Expected {expected_parents} staged parents, "
            f"found {len(parents)}"
        )

    for siblings in parents.values():
        validate_parent_children(siblings)

    if get_active_version(manifest_collection, docdb_id) != previous_version:
        raise RuntimeError("Active version changed before completion")

    record_completed_staging(
        manifest_collection=manifest_collection,
        docdb_id=docdb_id,
        version=version,
        expected_children=expected_children,
        expected_parents=expected_parents,
        previous_version=previous_version,
        child_ids=ids,
    )


def discard_incomplete_docdb_staging(
    child_collection: Any,
    manifest_collection: Any,
    docdb_id: str | int,
    version: str | int,
    *,
    batch_size: int = 100,
) -> int:
    """Discard an incomplete staged version without touching active children.

    Refuse cleanup if a completion record exists. Such records require
    explicit recovery or activation rather than automatic deletion.
    """
    docdb_id = str(docdb_id)
    version = int(version)

    if batch_size < 1:
        raise ValueError("Batch size must be positive")

    active = get_active_version(manifest_collection, docdb_id)

    if not validate_newer_version(version, active):
        raise ValueError(
            f"Cannot discard DocDB {docdb_id} v{version}; "
            f"active version is {active}"
        )

    completion = manifest_collection.get(
        ids=[staging_record_key(docdb_id, version)],
        include=["metadatas"],
    )

    if completion["ids"]:
        raise RuntimeError(
            "Completed staging exists; explicit recovery is required"
        )

    where = {
        "$and": [
            {"docdb_id": docdb_id},
            {"docdb_version": str(version)},
        ]
    }

    removed = 0

    while True:
        staged = child_collection.get(
            where=where,
            limit=batch_size,
            include=[],
        )

        ids = staged["ids"]

        if not ids:
            break

        if get_active_version(manifest_collection, docdb_id) != active:
            raise RuntimeError(
                "Active version changed during incomplete-stage cleanup"
            )

        child_collection.delete(ids=ids)
        removed += len(ids)

    if get_active_version(manifest_collection, docdb_id) != active:
        raise RuntimeError(
            "Active version changed during incomplete-stage cleanup"
        )

    remaining = child_collection.get(
        where=where,
        limit=1,
        include=[],
    )

    if remaining["ids"]:
        raise RuntimeError("Incomplete staged children remain")

    return removed
