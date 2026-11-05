"""Merge and rerank child-level retrieval candidates."""

from typing import Any


def merge_child_candidates(
    semantic_candidates: list[dict[str, Any]],
    keyword_candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Deduplicate children while retaining retrieval provenance."""

    merged = {}

    for method, candidates in (
        ("semantic", semantic_candidates),
        ("keyword", keyword_candidates),
    ):
        for candidate in candidates:
            actual_method = candidate.get("retrieval_method") or method
            source = str(candidate.get("source") or (
                "docdb" if "docdb_id" in candidate else
                "indico" if "indico_id" in candidate else "unknown"
            ))
            child_id = (source, str(candidate["document_id"]))

            if child_id not in merged:
                merged[child_id] = {
                    **candidate,
                    "retrieval_methods": [actual_method],
                }
            elif actual_method not in merged[child_id]["retrieval_methods"]:
                merged[child_id]["retrieval_methods"].append(actual_method)

            for score_key in ("semantic_distance", "bm25_score"):
                if score_key in candidate:
                    merged[child_id][score_key] = candidate[score_key]

    return list(merged.values())


def rerank_child_candidates(
    query: str,
    candidates: list[dict[str, Any]],
    reranker: Any,
    *,
    top_k: int | None = None,
    batch_size: int = 16,
) -> list[dict[str, Any]]:
    """Cross-encoder reranking of short child passages."""

    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive")

    if not candidates:
        return []

    tokenizer = reranker.tokenizer
    max_length = int(reranker.max_length)

    pairs = []

    for candidate in candidates:
        text = str(candidate["cleaned_text"])

        # Truncate the passage as the second sequence of a query/passage
        # pair, preserving the query whenever it fits.
        encoded = tokenizer(
            query,
            text,
            truncation="only_second",
            max_length=max_length,
            return_tensors=None,
        )

        passage_ids = tokenizer(
            text,
            add_special_tokens=False,
        )["input_ids"]

        # Recover the passage portion without assuming a particular
        # tokenizer's special-token positions.
        query_ids = tokenizer(
            query,
            add_special_tokens=False,
        )["input_ids"]

        special_tokens = tokenizer.num_special_tokens_to_add(pair=True)
        passage_budget = max_length - len(query_ids) - special_tokens

        if passage_budget <= 0:
            raise ValueError(
                "Query exceeds the cross-encoder input capacity"
            )

        truncated_text = tokenizer.decode(
            passage_ids[:passage_budget],
            skip_special_tokens=True,
        )

        pairs.append((query, truncated_text))

    scores = reranker.predict(pairs, batch_size=batch_size)

    ranked = [
        {
            **candidate,
            "reranker_score": float(score),
        }
        for candidate, score in zip(candidates, scores)
    ]

    ranked.sort(key=lambda item: item["reranker_score"], reverse=True)

    return ranked if top_k is None else ranked[:top_k]
