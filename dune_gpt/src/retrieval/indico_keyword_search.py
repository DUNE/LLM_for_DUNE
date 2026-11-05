"""BM25 retrieval over completed Indico child chunks."""

from threading import RLock
from typing import Any

from rank_bm25 import BM25Okapi

from src.retrieval.child_keyword_search import tokenize
from src.retrieval.metadata_filters import matches_metadata


class IndicoBM25Cache:
    """Thread-safe, reusable BM25 index for completed Indico children."""

    def __init__(self):
        self._lock = RLock()
        self._signature = None
        self._records = []
        self._bm25 = None

    def invalidate(self):
        """Call after Indico indexing updates to force a refresh."""
        with self._lock:
            self._signature = None

    def _refresh(self, child_collection):
        signature = child_collection.count()

        if signature == self._signature:
            return

        records = []
        offset = 0
        batch_size = 500

        while True:
            batch = child_collection.get(
                where={"index_status": "complete"},
                limit=batch_size,
                offset=offset,
                include=["documents", "metadatas"],
            )

            for child_id, document, metadata in zip(
                batch["ids"], batch["documents"], batch["metadatas"]
            ):
                records.append({
                    **metadata,
                    "document_id": child_id,
                    "cleaned_text": document,
                })

            if len(batch["ids"]) < batch_size:
                break

            offset += batch_size

        self._records = records
        self._bm25 = (
            BM25Okapi([tokenize(r["cleaned_text"]) for r in records])
            if records else None
        )
        self._signature = signature

    def search(
        self,
        child_collection: Any,
        query: str,
        *,
        candidate_count: int = 30,
        person: str | None = None,
        title: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[dict[str, Any]]:
        """Rank eligible Indico child chunks using BM25."""

        tokens = tokenize(query)
        if not tokens or candidate_count <= 0:
            return []

        with self._lock:
            self._refresh(child_collection)

            if self._bm25 is None:
                return []

            scores = self._bm25.get_scores(tokens)
            ranked = sorted(
                enumerate(scores),
                key=lambda pair: pair[1],
                reverse=True,
            )

            results = []

            for index, score in ranked:
                if score <= 0:
                    break

                record = self._records[index]

                if not matches_metadata(
                    record,
                    source="indico",
                    person=person,
                    title=title,
                    date_from=date_from,
                    date_to=date_to,
                ):
                    continue

                results.append({
                    **record,
                    "bm25_score": float(score),
                    "retrieval_method": "keyword",
                })

                if len(results) >= candidate_count:
                    break

            return results


_DEFAULT_CACHE = IndicoBM25Cache()


def search_indico_children_bm25(
    child_collection: Any,
    query: str,
    *,
    candidate_count: int = 30,
    person: str | None = None,
    title: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    cache: IndicoBM25Cache | None = None,
) -> list[dict[str, Any]]:
    """Convenience wrapper for cached Indico keyword retrieval."""

    selected_cache = cache if cache is not None else _DEFAULT_CACHE

    return selected_cache.search(
        child_collection,
        query,
        candidate_count=candidate_count,
        person=person,
        title=title,
        date_from=date_from,
        date_to=date_to,
    )
