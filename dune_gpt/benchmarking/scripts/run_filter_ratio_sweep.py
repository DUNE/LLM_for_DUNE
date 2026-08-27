#!/usr/bin/env python3
"""Sweep final slides/document filter ratios for fixed retriever pipelines."""

from __future__ import annotations

import argparse
import csv
import json
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
from retrieval_scores import ERS_FORMULA, compute_ers
from retriever_doc_type_filter import (
    DocumentTypeRatio,
    apply_document_type_ratio,
    parse_document_type_ratio,
    ratio_manifest,
)
from run_retriever_group_a import DEFAULT_DATA_PATH, DEFAULT_DOC_PREFIX, DEFAULT_MODEL, DEFAULT_QUERY_PREFIX
from run_retriever_group_a import FixedIndexRetriever
from run_retriever_group_b import RERANKER_MODELS, metric_row, rerank_cached_batch


DEFAULT_QA = "benchmarking/qa_sets/retrieval_qa_paired_v2_candidates.csv"
DEFAULT_RUN_NAME = "filter_ratio_sweep_paired_v2"
DEFAULT_RERANKER = "BAAI/bge-reranker-v2-m3"
RATIOS = [(slides, 6 - slides) for slides in range(6, -1, -1)]


def ratio_name(slides: int, documents: int) -> str:
    return f"slides{slides}_document{documents}"


def make_ratio(slides: int, documents: int) -> DocumentTypeRatio:
    return parse_document_type_ratio(f"slides:{slides},document:{documents}")


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def evaluate_ratio_rows(
    qa_rows: List[Dict[str, Any]],
    retrieved_rows: List[Dict[str, Any]],
    ratio: DocumentTypeRatio,
    filter_pool_top_k: int,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    rows = []
    for item in retrieved_rows:
        qa = item["qa"]
        top_ranked = [record.copy() for record in item["ranked"][:filter_pool_top_k]]
        retrieved = apply_document_type_ratio(top_ranked, ratio)
        score = score_retrieved(qa, retrieved, args.overlap_threshold)
        row_args = argparse.Namespace(top_k=ratio.final_k, overlap_threshold=args.overlap_threshold)
        rows.append(build_result_row(qa, retrieved, score, item["latency_ms"], row_args))
    if len(rows) != len(qa_rows):
        raise RuntimeError(f"Expected {len(qa_rows)} rows, got {len(rows)}")
    return rows


def summary_for_rows(
    rows: List[Dict[str, Any]],
    retriever: FixedIndexRetriever,
    ratio: DocumentTypeRatio,
    pipeline: str,
    filter_pool_top_k: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    summary_args = argparse.Namespace(
        qa=args.qa,
        data_path=args.data_path,
        embedding_model=args.embedding_model,
        top_k=ratio.final_k,
        overlap_threshold=args.overlap_threshold,
        document_prefix=args.document_prefix,
        query_prefix=args.query_prefix,
    )
    summary = aggregate(rows, summary_args, retriever.collection_count)
    summary.update(
        {
            "pipeline": pipeline,
            "hybrid_weight_dense": args.hybrid_weight_dense,
            "filter_pool_top_k": filter_pool_top_k,
            "final_top_k": ratio.final_k,
            "document_type_ratio_filter": ratio_manifest(ratio),
            "ERS_formula": ERS_FORMULA,
        }
    )
    summary["ERS"] = compute_ers(metric_row(summary))
    return summary


def table_row(
    pipeline: str,
    ratio: DocumentTypeRatio,
    slides: int,
    documents: int,
    filter_pool_top_k: int,
    summary: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    row = {
        "pipeline": pipeline,
        "ratio": ratio.raw,
        "slides_quota": slides,
        "document_quota": documents,
        "filter_pool_top_k": filter_pool_top_k,
        "final_top_k": ratio.final_k,
        "hybrid_weight_dense": args.hybrid_weight_dense,
        "path_at_k": summary["path_hit_at_k"],
        "anchor_at_k": summary["anchor_hit_at_k"],
        "mrr_path": summary["mrr_path"],
        "mrr_anchor": summary["mrr_anchor"],
        "best_ov": summary["mean_best_anchor_overlap"],
        "lat_ms": summary["mean_latency_ms"],
        "ERS": summary["ERS"],
    }
    if pipeline == "with_reranker":
        row.update(
            {
                "reranker_model": args.reranker_model,
                "candidate_k": args.reranker_candidate_k,
            }
        )
    else:
        row.update({"reranker_model": "", "candidate_k": ""})
    return row


def write_table(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "pipeline",
        "ratio",
        "slides_quota",
        "document_quota",
        "candidate_k",
        "filter_pool_top_k",
        "final_top_k",
        "hybrid_weight_dense",
        "reranker_model",
        "path_at_k",
        "anchor_at_k",
        "mrr_path",
        "mrr_anchor",
        "best_ov",
        "lat_ms",
        "ERS",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            formatted = row.copy()
            for key in ["path_at_k", "anchor_at_k", "mrr_path", "mrr_anchor", "best_ov", "lat_ms", "ERS"]:
                formatted[key] = f"{float(formatted[key]):.6f}"
            writer.writerow(formatted)


def run(args: argparse.Namespace) -> None:
    run_dir = resolve_path(args.output_root) / args.run_name
    if run_dir.exists() and any(run_dir.iterdir()) and not args.overwrite:
        raise RuntimeError(f"Run directory already exists and is not empty: {run_dir}. Use --overwrite.")

    qa_rows = load_qa_rows(resolve_path(args.qa))
    if args.limit is not None:
        qa_rows = qa_rows[: args.limit]

    neutral_ratio = parse_document_type_ratio("")
    args.document_type_ratio = neutral_ratio
    retriever = FixedIndexRetriever(args)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "qa_path": str(resolve_path(args.qa)),
        "data_path": str(resolve_path(args.data_path)),
        "embedding_model": args.embedding_model,
        "document_prefix": args.document_prefix,
        "query_prefix": args.query_prefix,
        "overlap_threshold": args.overlap_threshold,
        "score": {"ERS": ERS_FORMULA},
        "qa_count": len(qa_rows),
        "ratios": [f"slides:{slides},document:{documents}" for slides, documents in RATIOS],
        "without_reranker": {
            "base_retriever": "hybrid",
            "hybrid_weight_dense": args.hybrid_weight_dense,
            "filter_pool_top_k": args.no_reranker_top_k,
        },
        "with_reranker": {
            "base_retriever": "hybrid",
            "hybrid_weight_dense": args.hybrid_weight_dense,
            "reranker_model": args.reranker_model,
            "candidate_k": args.reranker_candidate_k,
            "filter_pool_top_k": args.reranker_top_k,
        },
    }
    write_json(run_dir / "run_manifest.json", manifest)

    started = time.perf_counter()
    print(f"Preparing no-reranker hybrid top{args.no_reranker_top_k} for {len(qa_rows)} QA rows", flush=True)
    no_reranker_rows = []
    for qa in qa_rows:
        query_started = time.perf_counter()
        ranked = retriever.retrieve(
            qa["question"],
            "hybrid",
            args.no_reranker_top_k,
            args.hybrid_weight_dense,
            apply_type_filter=False,
        )
        no_reranker_rows.append(
            {
                "qa": qa,
                "ranked": ranked,
                "latency_ms": (time.perf_counter() - query_started) * 1000,
            }
        )

    print(
        f"Preparing reranker candidates: hybrid w={args.hybrid_weight_dense:.2f}, "
        f"candidate_k={args.reranker_candidate_k}",
        flush=True,
    )
    candidate_rows = []
    for qa in qa_rows:
        query_started = time.perf_counter()
        candidates = retriever.retrieve(
            qa["question"],
            "hybrid",
            args.reranker_candidate_k,
            args.hybrid_weight_dense,
            apply_type_filter=False,
        )
        candidate_rows.append(
            {
                "qa": qa,
                "candidates": candidates,
                "candidate_latency_ms": (time.perf_counter() - query_started) * 1000,
            }
        )

    print(f"Loading reranker: {args.reranker_model}", flush=True)
    reranker = CrossEncoder(args.reranker_model, max_length=args.reranker_max_length)
    rerank_started = time.perf_counter()
    reranked_raw = rerank_cached_batch(reranker, retriever, candidate_rows, args.reranker_batch_size)
    rerank_elapsed_ms = (time.perf_counter() - rerank_started) * 1000
    per_query_rerank_ms = rerank_elapsed_ms / max(1, len(reranked_raw))
    with_reranker_rows = []
    for item in reranked_raw:
        with_reranker_rows.append(
            {
                "qa": item["qa"],
                "ranked": item["reranked"],
                "latency_ms": item["latency_ms"] + per_query_rerank_ms,
            }
        )
    print(f"Reranked {len(with_reranker_rows)} rows in {rerank_elapsed_ms / 1000:.2f}s", flush=True)

    table_rows: List[Dict[str, Any]] = []
    for slides, documents in RATIOS:
        ratio = make_ratio(slides, documents)
        name = ratio_name(slides, documents)
        print(f"Scoring ratio {ratio.raw}", flush=True)

        no_rows = evaluate_ratio_rows(
            qa_rows,
            no_reranker_rows,
            ratio,
            args.no_reranker_top_k,
            args,
        )
        no_summary = summary_for_rows(
            no_rows,
            retriever,
            ratio,
            "without_reranker",
            args.no_reranker_top_k,
            args,
        )
        no_dir = run_dir / "without_reranker" / name
        write_result_rows(no_rows, no_dir / "retrieval_eval.csv")
        write_summary(no_summary, no_dir / "retrieval_eval_summary.json")
        table_rows.append(
            table_row("without_reranker", ratio, slides, documents, args.no_reranker_top_k, no_summary, args)
        )

        rerank_rows = evaluate_ratio_rows(
            qa_rows,
            with_reranker_rows,
            ratio,
            args.reranker_top_k,
            args,
        )
        rerank_summary = summary_for_rows(
            rerank_rows,
            retriever,
            ratio,
            "with_reranker",
            args.reranker_top_k,
            args,
        )
        rerank_summary.update(
            {
                "reranker_model": args.reranker_model,
                "candidate_k": args.reranker_candidate_k,
                "reranker_max_length": args.reranker_max_length,
            }
        )
        rerank_dir = run_dir / "with_reranker" / name
        write_result_rows(rerank_rows, rerank_dir / "retrieval_eval.csv")
        write_summary(rerank_summary, rerank_dir / "retrieval_eval_summary.json")
        table_rows.append(
            table_row("with_reranker", ratio, slides, documents, args.reranker_top_k, rerank_summary, args)
        )

    table_path = run_dir / "filter_ratio_ers_table.csv"
    write_table(table_path, table_rows)
    write_json(
        run_dir / "timing_summary.json",
        {
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "qa_count": len(qa_rows),
            "ratio_count": len(RATIOS),
            "rerank_elapsed_seconds": round(rerank_elapsed_ms / 1000, 3),
        },
    )
    print(f"Run manifest: {run_dir / 'run_manifest.json'}")
    print(f"ERS table: {table_path}")
    print(f"Elapsed seconds: {time.perf_counter() - started:.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep slides/document final filter ratios.")
    parser.add_argument("--qa", default=DEFAULT_QA)
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--embedding-model", default=DEFAULT_MODEL)
    parser.add_argument("--document-prefix", default=DEFAULT_DOC_PREFIX)
    parser.add_argument("--query-prefix", default=DEFAULT_QUERY_PREFIX)
    parser.add_argument("--overlap-threshold", type=float, default=0.2)
    parser.add_argument("--output-root", default="benchmarking/retriever_runs")
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--hybrid-weight-dense", type=float, default=0.50)
    parser.add_argument("--no-reranker-top-k", type=int, default=10)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER, choices=RERANKER_MODELS)
    parser.add_argument("--reranker-candidate-k", type=int, default=15)
    parser.add_argument("--reranker-top-k", type=int, default=8)
    parser.add_argument("--reranker-max-length", type=int, default=512)
    parser.add_argument("--reranker-batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
