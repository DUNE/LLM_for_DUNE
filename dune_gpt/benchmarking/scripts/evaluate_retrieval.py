#!/usr/bin/env python3
"""
Evaluate retrieval quality for benchmark Chroma indexes.

This is a retrieval-only evaluator. It reads the fixed QA CSV, queries one
Chroma experiment database, and scores whether returned chunks match the
expected attachment and source anchor span.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import pysqlite3

    sys.modules["sqlite3"] = pysqlite3
except Exception:
    pass

import chromadb
from chromadb.config import Settings


ROOT = Path(__file__).resolve().parents[2]
COLLECTION_NAME = "DUNE_VECTOR_DB"


class BenchmarkEmbeddingFunction:
    """Chroma embedding function matching the local indexing wrapper."""

    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        document_prefix: str = "passage: ",
        query_prefix: str = "passage: ",
    ):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.document_prefix = document_prefix
        self.query_prefix = query_prefix

    def __call__(self, input: List[str]) -> List[List[float]]:
        return self.embed_query(input)

    def embed_documents(self, input: List[str]) -> List[List[float]]:
        if not isinstance(input, list):
            raise ValueError(f"Input must be a list, received {type(input)}")
        return self.model.encode(
            [f"{self.document_prefix}{text}" for text in input],
            normalize_embeddings=True,
        ).tolist()

    def embed_query(self, input: List[str] | str) -> List[List[float]]:
        if isinstance(input, str):
            texts = [input]
        elif isinstance(input, list):
            texts = input
        else:
            raise ValueError(f"Input must be a string or list, received {type(input)}")
        return self.model.encode(
            [f"{self.query_prefix}{text}" for text in texts],
            normalize_embeddings=True,
        ).tolist()

    def name(self) -> str:
        return self.model_name


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def normalize_local_path(path: str) -> str:
    return str(path or "").strip().replace("/", "\\").lower()


def parse_int(value: Any, default: int = -1) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def source_from_path(path: str) -> str:
    normalized = normalize_local_path(path)
    if "\\docdb\\" in normalized:
        return "docdb"
    if "\\indico\\" in normalized:
        return "indico"
    return "unknown"


def load_qa_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No QA rows found in {path}")

    required = {"id", "question", "expected_local_path"}
    missing = required - set(rows[0].keys())
    if missing:
        raise RuntimeError(f"QA file is missing required columns: {sorted(missing)}")
    return rows


def get_collection(
    data_path: Path,
    embedding_model: str,
    device: str | None,
    document_prefix: str,
    query_prefix: str,
):
    if not data_path.exists():
        raise RuntimeError(f"Chroma data path does not exist: {data_path}")

    client = chromadb.PersistentClient(path=str(data_path), settings=Settings())
    collections = [collection.name for collection in client.list_collections()]
    if COLLECTION_NAME not in collections:
        raise RuntimeError(
            f"Collection '{COLLECTION_NAME}' not found in {data_path}. "
            f"Available collections: {collections}"
        )

    embedding_function = BenchmarkEmbeddingFunction(
        embedding_model,
        device=device,
        document_prefix=document_prefix,
        query_prefix=query_prefix,
    )
    return client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_function)


def span_overlap(
    expected_start: int,
    expected_end: int,
    chunk_start: int,
    chunk_end: int,
) -> Tuple[int, float]:
    if min(expected_start, expected_end, chunk_start, chunk_end) < 0:
        return 0, 0.0
    expected_len = max(0, expected_end - expected_start)
    if expected_len == 0:
        return 0, 0.0
    overlap = max(0, min(expected_end, chunk_end) - max(expected_start, chunk_start))
    return overlap, overlap / expected_len


def score_retrieved(
    qa: Dict[str, Any],
    retrieved: List[Dict[str, Any]],
    overlap_threshold: float,
) -> Dict[str, Any]:
    expected_path = normalize_local_path(qa.get("expected_local_path", ""))
    anchor_word_start = parse_int(qa.get("anchor_start_word"))
    anchor_word_end = parse_int(qa.get("anchor_end_word"))
    anchor_char_start = parse_int(qa.get("anchor_start_char"))
    anchor_char_end = parse_int(qa.get("anchor_end_char"))

    path_hit_rank = 0
    anchor_hit_rank = 0
    best_anchor_overlap = 0.0
    best_anchor_overlap_rank = 0

    for item in retrieved:
        rank = item["rank"]
        metadata = item["metadata"]
        local_path = normalize_local_path(metadata.get("local_path", ""))
        if local_path != expected_path:
            continue

        if not path_hit_rank:
            path_hit_rank = rank

        word_overlap, word_ratio = span_overlap(
            anchor_word_start,
            anchor_word_end,
            parse_int(metadata.get("chunk_start_word")),
            parse_int(metadata.get("chunk_end_word")),
        )
        char_overlap, char_ratio = span_overlap(
            anchor_char_start,
            anchor_char_end,
            parse_int(metadata.get("chunk_start_char")),
            parse_int(metadata.get("chunk_end_char")),
        )
        overlap_ratio = word_ratio if word_overlap > 0 else char_ratio

        if overlap_ratio > best_anchor_overlap:
            best_anchor_overlap = overlap_ratio
            best_anchor_overlap_rank = rank
        if not anchor_hit_rank and overlap_ratio >= overlap_threshold:
            anchor_hit_rank = rank

    return {
        "path_hit": bool(path_hit_rank),
        "path_hit_rank": path_hit_rank,
        "anchor_hit": bool(anchor_hit_rank),
        "anchor_hit_rank": anchor_hit_rank,
        "best_anchor_overlap": best_anchor_overlap,
        "best_anchor_overlap_rank": best_anchor_overlap_rank,
    }


def simplify_retrieved(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    retrieved = []
    for index, doc_id in enumerate(ids):
        metadata = metadatas[index] or {}
        text = documents[index] or ""
        distance = distances[index] if index < len(distances) else None
        retrieved.append(
            {
                "rank": index + 1,
                "document_id": doc_id,
                "distance": distance,
                "metadata": metadata,
                "text_preview": text[:500].replace("\n", " "),
            }
        )
    return retrieved


def query_one(collection, question: str, top_k: int) -> Tuple[List[Dict[str, Any]], float]:
    started = time.perf_counter()
    results = collection.query(
        query_texts=[question],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    latency_ms = (time.perf_counter() - started) * 1000
    return simplify_retrieved(results), latency_ms


def reciprocal_rank(rank: int) -> float:
    return 0.0 if not rank else 1.0 / rank


def aggregate(rows: List[Dict[str, Any]], args: argparse.Namespace, collection_count: int) -> Dict[str, Any]:
    def summarize(items: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not items:
            return {
                "count": 0,
                "path_hit_at_k": 0.0,
                "anchor_hit_at_k": 0.0,
                "mrr_path": 0.0,
                "mrr_anchor": 0.0,
                "mean_best_anchor_overlap": 0.0,
                "mean_latency_ms": 0.0,
            }
        return {
            "count": len(items),
            "path_hit_at_k": sum(1 for row in items if row["path_hit"]) / len(items),
            "anchor_hit_at_k": sum(1 for row in items if row["anchor_hit"]) / len(items),
            "mrr_path": sum(reciprocal_rank(row["path_hit_rank"]) for row in items) / len(items),
            "mrr_anchor": sum(reciprocal_rank(row["anchor_hit_rank"]) for row in items) / len(items),
            "mean_best_anchor_overlap": sum(row["best_anchor_overlap"] for row in items) / len(items),
            "mean_latency_ms": statistics.mean(row["latency_ms"] for row in items),
        }

    by_source: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[row["expected_source"]].append(row)
        by_type[row["question_type"] or "unknown"].append(row)

    return {
        "qa_path": str(resolve_path(args.qa)),
        "data_path": str(resolve_path(args.data_path)),
        "embedding_model": args.embedding_model,
        "collection_name": COLLECTION_NAME,
        "collection_count": collection_count,
        "top_k": args.top_k,
        "overlap_threshold": args.overlap_threshold,
        "document_prefix": args.document_prefix,
        "query_prefix": args.query_prefix,
        **summarize(rows),
        "by_source": {key: summarize(value) for key, value in sorted(by_source.items())},
        "by_question_type": {key: summarize(value) for key, value in sorted(by_type.items())},
    }


def write_result_rows(rows: List[Dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "question",
        "question_type",
        "expected_source",
        "expected_local_path",
        "top_k",
        "path_hit",
        "path_hit_rank",
        "anchor_hit",
        "anchor_hit_rank",
        "best_anchor_overlap",
        "best_anchor_overlap_rank",
        "latency_ms",
        "top1_document_id",
        "top1_local_path",
        "top1_distance",
        "top1_overlap",
        "retrieved_json",
    ]
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(summary: Dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def build_result_row(
    qa: Dict[str, Any],
    retrieved: List[Dict[str, Any]],
    score: Dict[str, Any],
    latency_ms: float,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    top1 = retrieved[0] if retrieved else {"metadata": {}, "document_id": "", "distance": None}
    top1_metadata = top1.get("metadata", {})
    top1_score = score_retrieved(qa, [top1], args.overlap_threshold) if retrieved else {"best_anchor_overlap": 0.0}

    compact_retrieved = []
    for item in retrieved:
        metadata = item["metadata"]
        compact_retrieved.append(
            {
                "rank": item["rank"],
                "document_id": item["document_id"],
                "distance": item["distance"],
                "local_path": metadata.get("local_path", ""),
                "document_type": metadata.get("document_type", ""),
                "attachment_url": metadata.get("attachment_url", ""),
                "chunk_start_word": metadata.get("chunk_start_word", ""),
                "chunk_end_word": metadata.get("chunk_end_word", ""),
                "chunk_start_char": metadata.get("chunk_start_char", ""),
                "chunk_end_char": metadata.get("chunk_end_char", ""),
                "text_preview": item["text_preview"],
            }
        )

    return {
        "id": qa.get("id", ""),
        "question": qa.get("question", ""),
        "question_type": qa.get("question_type", ""),
        "expected_source": source_from_path(qa.get("expected_local_path", "")),
        "expected_local_path": qa.get("expected_local_path", ""),
        "top_k": args.top_k,
        "path_hit": score["path_hit"],
        "path_hit_rank": score["path_hit_rank"],
        "anchor_hit": score["anchor_hit"],
        "anchor_hit_rank": score["anchor_hit_rank"],
        "best_anchor_overlap": round(score["best_anchor_overlap"], 6),
        "best_anchor_overlap_rank": score["best_anchor_overlap_rank"],
        "latency_ms": round(latency_ms, 3),
        "top1_document_id": top1.get("document_id", ""),
        "top1_local_path": top1_metadata.get("local_path", ""),
        "top1_distance": top1.get("distance", ""),
        "top1_overlap": round(top1_score["best_anchor_overlap"], 6),
        "retrieved_json": json.dumps(compact_retrieved, ensure_ascii=False),
    }


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    qa_path = resolve_path(args.qa)
    data_path = resolve_path(args.data_path)
    output_path = resolve_path(args.output)
    summary_path = resolve_path(args.summary_output)

    qa_rows = load_qa_rows(qa_path)
    if args.limit is not None:
        qa_rows = qa_rows[: args.limit]

    collection = get_collection(
        data_path,
        args.embedding_model,
        args.device,
        args.document_prefix,
        args.query_prefix,
    )
    collection_count = collection.count()
    if collection_count == 0:
        raise RuntimeError(f"Collection '{COLLECTION_NAME}' is empty in {data_path}")

    result_rows = []
    for index, qa in enumerate(qa_rows, start=1):
        retrieved, latency_ms = query_one(collection, qa["question"], args.top_k)
        score = score_retrieved(qa, retrieved, args.overlap_threshold)
        result_rows.append(build_result_row(qa, retrieved, score, latency_ms, args))
        if args.progress_every and index % args.progress_every == 0:
            print(f"Evaluated {index}/{len(qa_rows)} questions")

    summary = aggregate(result_rows, args, collection_count)
    write_result_rows(result_rows, output_path)
    write_summary(summary, summary_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality for a Chroma benchmark index.")
    parser.add_argument("--qa", default="benchmarking/qa_sets/retrieval_qa.csv")
    parser.add_argument("--data-path", required=True, help="Path to a Chroma experiment directory.")
    parser.add_argument("--embedding-model", default="multi-qa-mpnet-base-dot-v1")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--overlap-threshold", type=float, default=0.2)
    parser.add_argument("--output", default="benchmarking/results/retrieval_eval.csv")
    parser.add_argument("--summary-output", default="benchmarking/results/retrieval_eval_summary.json")
    parser.add_argument("--limit", type=int, default=None, help="Optional QA row limit for smoke tests.")
    parser.add_argument("--device", default=None, help="SentenceTransformers device override, e.g. cpu or cuda.")
    parser.add_argument("--document-prefix", default="passage: ")
    parser.add_argument("--query-prefix", default="passage: ")
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = evaluate(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
