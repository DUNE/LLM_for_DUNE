#!/usr/bin/env python3
"""Run retriever-layer Group B hybrid reranker matrix on a fixed Chroma index."""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from sentence_transformers import CrossEncoder

from evaluate_retrieval import (
    aggregate,
    build_result_row,
    load_qa_rows,
    resolve_path,
    score_retrieved,
    write_result_rows,
    write_summary,
)
from retrieval_scores import ERS_FORMULA, RERS_FORMULA, compute_ers, compute_rers
from retriever_doc_type_filter import apply_document_type_ratio, document_type_ratio_from_env, ratio_manifest
from run_retriever_group_a import FixedIndexRetriever


DEFAULT_DATA_PATH = "benchmarking/chroma_experiments/intfloat_e5-small-v2_word_chunk2000_overlap0_both"
DEFAULT_MODEL = "intfloat/e5-small-v2"
DEFAULT_DOC_PREFIX = "passage: "
DEFAULT_QUERY_PREFIX = "query: "
DEFAULT_RUN_NAME = "e5_small_v2_2000_overlap0_group_b_hybrid_reranker"
CANDIDATE_K = 15
FINAL_TOP_K_VALUES = [3, 5, 10]
HYBRID_WEIGHTS = [0.25, 0.50, 0.75]
RERANKER_MODELS = [
    "cross-encoder/ms-marco-MiniLM-L6-v2",
    "BAAI/bge-reranker-base",
    "BAAI/bge-reranker-v2-m3",
]


def slugify(value: str) -> str:
    value = value.replace("/", "_")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def metric_row(summary: Dict[str, Any]) -> Dict[str, float]:
    return {
        "path_at_k": float(summary["path_hit_at_k"]),
        "anchor_at_k": float(summary["anchor_hit_at_k"]),
        "mrr_path": float(summary["mrr_path"]),
        "mrr_anchor": float(summary["mrr_anchor"]),
        "best_ov": float(summary["mean_best_anchor_overlap"]),
    }


def combo_records(args: argparse.Namespace) -> List[Dict[str, Any]]:
    combos = []
    for reranker_model in args.reranker_models:
        reranker_slug = slugify(reranker_model)
        for weight in HYBRID_WEIGHTS:
            weight_name = f"{int(weight * 100):03d}"
            for top_k in args.final_top_k_values:
                combos.append(
                    {
                        "name": f"hybrid_rerank_{reranker_slug}_w{weight_name}_cand{args.candidate_k}_top{top_k}",
                        "base_retriever": "hybrid",
                        "candidate_k": args.candidate_k,
                        "final_top_k": top_k,
                        "hybrid_weight_dense": weight,
                        "reranker_model": reranker_model,
                    }
                )
    return combos


def manifest(args: argparse.Namespace, combos: List[Dict[str, Any]], run_dir: Path) -> Dict[str, Any]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "qa_path": str(resolve_path(args.qa)),
        "data_path": str(resolve_path(args.data_path)),
        "embedding_model": args.embedding_model,
        "document_prefix": args.document_prefix,
        "query_prefix": args.query_prefix,
        "overlap_threshold": args.overlap_threshold,
        "group": "B_hybrid_reranker",
        "combo_count": len(combos),
        "candidate_k": args.candidate_k,
        "combos": combos,
        "score": {
            "ERS": ERS_FORMULA,
            "RERS": RERS_FORMULA,
        },
        "document_type_ratio_filter": ratio_manifest(args.document_type_ratio),
    }


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def append_jsonl(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(data, ensure_ascii=False) + "\n")


def rerank_candidates(
    reranker: CrossEncoder,
    retriever: FixedIndexRetriever,
    question: str,
    candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    pairs = [
        (question, retriever.id_to_record[candidate["document_id"]]["document"])
        for candidate in candidates
    ]
    scores = reranker.predict(pairs)
    ranked = []
    for candidate, score in sorted(zip(candidates, scores), key=lambda item: float(item[1]), reverse=True):
        item = candidate.copy()
        item["reranker_score"] = float(score)
        item["distance"] = float(score)
        ranked.append(item)
    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank
    return ranked


def rerank_cached_batch(
    reranker: CrossEncoder,
    retriever: FixedIndexRetriever,
    cached_rows: List[Dict[str, Any]],
    batch_size: int,
) -> List[Dict[str, Any]]:
    pairs = []
    offsets = []
    for cached in cached_rows:
        start = len(pairs)
        qa = cached["qa"]
        for candidate in cached["candidates"]:
            pairs.append((qa["question"], retriever.id_to_record[candidate["document_id"]]["document"]))
        offsets.append((start, len(pairs), cached))

    scores = reranker.predict(pairs, batch_size=batch_size)
    reranked_rows = []
    for start, end, cached in offsets:
        candidate_scores = scores[start:end]
        ranked = []
        for candidate, score in sorted(
            zip(cached["candidates"], candidate_scores),
            key=lambda item: float(item[1]),
            reverse=True,
        ):
            item = candidate.copy()
            item["reranker_score"] = float(score)
            item["distance"] = float(score)
            ranked.append(item)
        for rank, item in enumerate(ranked, start=1):
            item["rank"] = rank
        reranked_rows.append(
            {
                "qa": cached["qa"],
                "reranked": ranked,
                "latency_ms": cached["candidate_latency_ms"],
            }
        )
    return reranked_rows


def write_ranked_csv(path: Path, combo_results: List[Dict[str, Any]]) -> None:
    rows = []
    for result in combo_results:
        summary = result["summary"]
        combo = result["combo"]
        rows.append(
            {
                "combo_name": combo["name"],
                "base_retriever": combo["base_retriever"],
                "reranker_model": combo["reranker_model"],
                "candidate_k": combo["candidate_k"],
                "rerank_top_k": combo["final_top_k"],
                "final_top_k": result["filtered_result_k"],
                "hybrid_weight_dense": combo["hybrid_weight_dense"],
                "path_at_k": summary["path_hit_at_k"],
                "anchor_at_k": summary["anchor_hit_at_k"],
                "mrr_path": summary["mrr_path"],
                "mrr_anchor": summary["mrr_anchor"],
                "best_ov": summary["mean_best_anchor_overlap"],
                "lat_ms": summary["mean_latency_ms"],
                "ERS": result["ERS"],
                "RERS": result["RERS"],
            }
        )
    rows.sort(key=lambda row: row["RERS"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    fieldnames = [
        "rank",
        "combo_name",
        "base_retriever",
        "reranker_model",
        "candidate_k",
        "rerank_top_k",
        "final_top_k",
        "hybrid_weight_dense",
        "path_at_k",
        "anchor_at_k",
        "mrr_path",
        "mrr_anchor",
        "best_ov",
        "lat_ms",
        "ERS",
        "RERS",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            formatted = row.copy()
            for key in ["path_at_k", "anchor_at_k", "mrr_path", "mrr_anchor", "best_ov", "lat_ms", "ERS", "RERS"]:
                formatted[key] = f"{float(formatted[key]):.6f}"
            writer.writerow(formatted)


def run(args: argparse.Namespace) -> None:
    args.document_type_ratio = document_type_ratio_from_env()
    run_dir = resolve_path(args.output_root) / args.run_name
    combos = combo_records(args)
    if args.document_type_ratio.enabled:
        too_small = [combo["final_top_k"] for combo in combos if combo["final_top_k"] < args.document_type_ratio.final_k]
        if too_small:
            raise ValueError(
                "When RETRIEVER_DOCUMENT_TYPE_RATIO is enabled, every reranked top-k "
                f"must be >= the quota sum ({args.document_type_ratio.final_k}). "
                f"Invalid top-k values: {sorted(set(too_small))}. "
                "Use --final-top-k-values with values large enough for the ratio."
            )
    if args.dry_run:
        print(json.dumps(manifest(args, combos, run_dir), indent=2, ensure_ascii=False))
        return

    write_json(run_dir / "run_manifest.json", manifest(args, combos, run_dir))
    results_path = run_dir / "combo_results.jsonl"
    if results_path.exists() and not args.append:
        results_path.unlink()

    qa_rows = load_qa_rows(resolve_path(args.qa))
    if args.limit is not None:
        qa_rows = qa_rows[: args.limit]

    retriever = FixedIndexRetriever(args)
    combo_results = []
    reranker_cache: Dict[str, CrossEncoder] = {}
    started_run = time.perf_counter()

    print(f"Precomputing hybrid candidate pools for {len(qa_rows)} QA rows")
    candidate_cache: Dict[float, List[Dict[str, Any]]] = {}
    for weight in HYBRID_WEIGHTS:
        weight_started = time.perf_counter()
        cached_rows = []
        for qa in qa_rows:
            query_started = time.perf_counter()
            candidates = retriever.retrieve(
                qa["question"],
                "hybrid",
                args.candidate_k,
                weight,
                apply_type_filter=False,
            )
            cached_rows.append(
                {
                    "qa": qa,
                    "candidates": candidates,
                    "candidate_latency_ms": (time.perf_counter() - query_started) * 1000,
                }
            )
        candidate_cache[weight] = cached_rows
        print(
            f"  w={weight:.2f}: cached {len(cached_rows)} candidate lists "
            f"in {time.perf_counter() - weight_started:.2f}s",
            flush=True,
        )

    combo_lookup = {(combo["reranker_model"], combo["hybrid_weight_dense"], combo["final_top_k"]): combo for combo in combos}
    result_lookup: Dict[str, Dict[str, Any]] = {}
    completed = 0

    for reranker_model in args.reranker_models:
        if reranker_model not in reranker_cache:
            print(f"  Loading reranker: {reranker_model}")
            reranker_cache[reranker_model] = CrossEncoder(reranker_model, max_length=args.reranker_max_length)
        reranker = reranker_cache[reranker_model]

        for weight in HYBRID_WEIGHTS:
            rerank_started = time.perf_counter()
            reranked_rows = rerank_cached_batch(
                reranker,
                retriever,
                candidate_cache[weight],
                args.reranker_batch_size,
            )
            rerank_elapsed_ms = (time.perf_counter() - rerank_started) * 1000
            per_query_rerank_ms = rerank_elapsed_ms / max(1, len(reranked_rows))
            for row in reranked_rows:
                row["latency_ms"] += per_query_rerank_ms
            print(
                f"  Reranked {len(reranked_rows)} QA rows with {reranker_model}, "
                f"w={weight:.2f} in {time.perf_counter() - rerank_started:.2f}s",
                flush=True,
            )

            for top_k in args.final_top_k_values:
                combo = combo_lookup[(reranker_model, weight, top_k)]
                completed += 1
                print(f"[{completed}/{len(combos)}] Scoring {combo['name']}")
                combo_dir = run_dir / combo["name"]
                output_csv = combo_dir / "retrieval_eval.csv"
                summary_json = combo_dir / "retrieval_eval_summary.json"
                started = time.perf_counter()
                rows = []
                filtered_result_k = args.document_type_ratio.final_k if args.document_type_ratio.enabled else top_k

                for item in reranked_rows:
                    qa = item["qa"]
                    top_ranked = [record.copy() for record in item["reranked"][:top_k]]
                    retrieved = apply_document_type_ratio(top_ranked, args.document_type_ratio)
                    for rank, record in enumerate(retrieved, start=1):
                        record["rank"] = rank
                    score = score_retrieved(qa, retrieved, args.overlap_threshold)
                    row_args = argparse.Namespace(top_k=filtered_result_k, overlap_threshold=args.overlap_threshold)
                    rows.append(build_result_row(qa, retrieved, score, item["latency_ms"], row_args))

                summary_args = argparse.Namespace(
                    qa=args.qa,
                    data_path=args.data_path,
                    embedding_model=args.embedding_model,
                    top_k=filtered_result_k,
                    overlap_threshold=args.overlap_threshold,
                    document_prefix=args.document_prefix,
                    query_prefix=args.query_prefix,
                )
                summary = aggregate(rows, summary_args, retriever.collection_count)
                summary.update(
                    {
                        "base_retriever": combo["base_retriever"],
                        "candidate_k": combo["candidate_k"],
                        "rerank_top_k": combo["final_top_k"],
                        "final_top_k": filtered_result_k,
                        "hybrid_weight_dense": combo["hybrid_weight_dense"],
                        "reranker_model": combo["reranker_model"],
                        "reranker_max_length": args.reranker_max_length,
                        "document_type_ratio_filter": ratio_manifest(args.document_type_ratio),
                    }
                )
                combo_ers = compute_ers(metric_row(summary))
                combo_rers = compute_rers(metric_row(summary), filtered_result_k)
                summary["ERS"] = combo_ers
                summary["RERS"] = combo_rers

                write_result_rows(rows, output_csv)
                write_summary(summary, summary_json)
                result = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "combo": combo,
                    "status": "success",
                    "combo_dir": str(combo_dir),
                    "result_csv": str(output_csv),
                    "result_summary": str(summary_json),
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                    "ERS": combo_ers,
                    "RERS": combo_rers,
                    "filtered_result_k": filtered_result_k,
                    "summary": summary,
                }
                append_jsonl(results_path, result)
                combo_results.append(result)
                result_lookup[combo["name"]] = result
                print(
                    "  Completed "
                    f"ERS={combo_ers:.4f} RERS={combo_rers:.4f} "
                    f"path@k={summary['path_hit_at_k']:.4f} "
                    f"anchor@k={summary['anchor_hit_at_k']:.4f}"
                )

    ranked_csv = run_dir / "retriever_group_b_rers_ranked.csv"
    write_ranked_csv(ranked_csv, combo_results)
    elapsed_run = time.perf_counter() - started_run
    write_json(
        run_dir / "timing_summary.json",
        {
            "elapsed_seconds": round(elapsed_run, 3),
            "qa_count": len(qa_rows),
            "combo_count": len(combos),
            "candidate_k": args.candidate_k,
            "reranker_model_count": len(args.reranker_models),
            "hybrid_weight_count": len(HYBRID_WEIGHTS),
            "final_top_k_values": args.final_top_k_values,
        },
    )
    print(f"Run manifest: {run_dir / 'run_manifest.json'}")
    print(f"Per-combo results: {results_path}")
    print(f"Ranked CSV: {ranked_csv}")
    print(f"Elapsed seconds: {elapsed_run:.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run retriever-layer Group B hybrid reranker matrix.")
    parser.add_argument("--qa", default="benchmarking/qa_sets/retrieval_qa_paired_candidates.csv")
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--embedding-model", default=DEFAULT_MODEL)
    parser.add_argument("--document-prefix", default=DEFAULT_DOC_PREFIX)
    parser.add_argument("--query-prefix", default=DEFAULT_QUERY_PREFIX)
    parser.add_argument("--overlap-threshold", type=float, default=0.2)
    parser.add_argument("--output-root", default="benchmarking/retriever_runs")
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--candidate-k", type=int, default=CANDIDATE_K)
    parser.add_argument("--reranker-models", nargs="+", default=RERANKER_MODELS)
    parser.add_argument("--reranker-max-length", type=int, default=512)
    parser.add_argument("--reranker-batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--final-top-k-values", type=int, nargs="+", default=FINAL_TOP_K_VALUES)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--append", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
