"""BM25 keyword retrieval over active DocDB child chunks."""

import re
import threading
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from src.indexing.docdb_versioned_store import get_all_active_versions


def tokenize(text: str) -> list[str]:
    """Use consistent tokenization for documents and queries."""
    return re.findall(r"\w+", text.lower())


class DocDBBM25Cache:
    """Cache the active DocDB BM25 corpus between queries."""

    def __init__(self):
        self._lock = threading.RLock()
        self._signature = None
        self._candidates = []
        self._bm25 = None
        self.build_count = 0

    def search(
        self,
        child_collection: Any,
        manifest_collection: Any,
        query: str,
        *,
        candidate_count: int = 30,
        batch_size: int = 500,
    ) -> list[dict[str, Any]]:
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty")

        if candidate_count <= 0 or batch_size <= 0:
            raise ValueError("candidate_count and batch_size must be positive")

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        with self._lock:
            active_versions = get_all_active_versions(manifest_collection)
            signature = (
                tuple(sorted(active_versions.items())),
                child_collection.count(),
            )

            if signature != self._signature:
                candidates = []
                offset = 0

                while active_versions:
                    batch = child_collection.get(
                        limit=batch_size,
                        offset=offset,
                        include=["documents", "metadatas"],
                    )

                    if not batch["ids"]:
                        break

                    for record_id, text, metadata in zip(
                        batch["ids"],
                        batch["documents"],
                        batch["metadatas"],
                    ):
                        docdb_id = str(metadata.get("docdb_id", ""))
                        version = int(metadata.get("docdb_version", -1))

                        if active_versions.get(docdb_id) != version:
                            continue

                        if not text or not text.strip():
                            continue

                        candidates.append({
                            **metadata,
                            "document_id": record_id,
                            "cleaned_text": text,
                        })

                    offset += len(batch["ids"])

                self._candidates = candidates
                self._bm25 = (
                    BM25Okapi([
                        tokenize(item["cleaned_text"])
                        for item in candidates
                    ])
                    if candidates else None
                )

                self._signature = signature
                self.build_count += 1

            if self._bm25 is None:
                return []

            scores = self._bm25.get_scores(query_tokens)
            ranked = np.argsort(scores)[::-1]

            results = []

            for index in ranked:
                score = float(scores[index])

                if score <= 0:
                    continue

                results.append({
                    **self._candidates[index],
                    "bm25_score": score,
                    "retrieval_method": "keyword",
                })

                if len(results) >= candidate_count:
                    break

            return results


_DEFAULT_BM25_CACHE = DocDBBM25Cache()


def search_docdb_children_bm25(
    child_collection: Any,
    manifest_collection: Any,
    query: str,
    *,
    candidate_count: int = 30,
    batch_size: int = 500,
) -> list[dict[str, Any]]:
    """Search active DocDB children using cached BM25."""

    return _DEFAULT_BM25_CACHE.search(
        child_collection,
        manifest_collection,
        query,
        candidate_count=candidate_count,
        batch_size=batch_size,
    )
