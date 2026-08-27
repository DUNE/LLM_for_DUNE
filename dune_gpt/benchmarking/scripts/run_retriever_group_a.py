#!/usr/bin/env python3
"""Run no-reranker retriever-layer matrix on a fixed Chroma index."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from rank_bm25 import BM25Okapi

from evaluate_retrieval import (
    COLLECTION_NAME,
    aggregate,
    build_result_row,
    get_collection,
    load_qa_rows,
    resolve_path,
    score_retrieved,
    write_result_rows,
    write_summary,
)
from retrieval_scores import ERS_FORMULA, RERS_FORMULA, compute_ers, compute_rers
from retriever_doc_type_filter import (
    apply_document_type_ratio,
    document_type_ratio_from_env,
    ratio_manifest,
)


DEFAULT_DATA_PATH = "benchmarking/chroma_experiments/intfloat_e5-small-v2_word_chunk2000_overlap0_both"
DEFAULT_MODEL = "intfloat/e5-small-v2"
DEFAULT_DOC_PREFIX = "passage: "
DEFAULT_QUERY_PREFIX = "query: "
DEFAULT_RUN_NAME = "e5_small_v2_2000_overlap0_group_a_no_reranker"
FINAL_TOP_K_VALUES = [3, 5, 10]
HYBRID_WEIGHTS = [0.25, 0.50, 0.75]


def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", str(text).lower())


def normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    values = list(scores.values())
    low = min(values)
    high = max(values)
    if math.isclose(high, low):
        return {key: 1.0 for key in scores}
    return {key: (value - low) / (high - low) for key, value in scores.items()}


def reciprocal_rank(rank: int) -> float:
    return 0.0 if not rank else 1.0 / rank


def ers(summary: Dict[str, Any]) -> float:
    return compute_ers(
        {
            "path_at_k": summary["path_hit_at_k"],
            "anchor_at_k": summary["anchor_hit_at_k"],
            "mrr_path": summary["mrr_path"],
            "mrr_anchor": summary["mrr_anchor"],
            "best_ov": summary["mean_best_anchor_overlap"],
        }
    )


def rers(summary: Dict[str, Any], final_top_k: int) -> float:
    return compute_rers(
        {
            "path_at_k": summary["path_hit_at_k"],
            "anchor_at_k": summary["anchor_hit_at_k"],
            "mrr_path": summary["mrr_path"],
            "mrr_anchor": summary["mrr_anchor"],
            "best_ov": summary["mean_best_anchor_overlap"],
        },
        final_top_k,
    )


class FixedIndexRetriever:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.collection = get_collection(
            resolve_path(args.data_path),
            args.embedding_model,
            args.device,
            args.document_prefix,
            args.query_prefix,
        )
        self.collection_count = self.collection.count()
        if self.collection_count == 0:
            raise RuntimeError(f"Collection '{COLLECTION_NAME}' is empty in {args.data_path}")

        all_records = self.collection.get(include=["documents", "metadatas"])
        self.ids = all_records["ids"]
        self.documents = all_records["documents"]
        self.metadatas = all_records["metadatas"]
        self.id_to_record = {
            doc_id: {
                "document_id": doc_id,
                "document": document or "",
                "metadata": metadata or {},
            }
            for doc_id, document, metadata in zip(self.ids, self.documents, self.metadatas)
        }
        self.bm25 = BM25Okapi([tokenize(document) for document in self.documents])

    def dense_scores(self, question: str) -> Dict[str, float]:
        results = self.collection.query(
            query_texts=[question],
            n_results=self.collection_count,
            include=["documents", "metadatas", "distances"],
        )
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        return {doc_id: 1.0 - float(distance) for doc_id, distance in zip(ids, distances)}

    def bm25_scores(self, question: str) -> Dict[str, float]:
        scores = self.bm25.get_scores(tokenize(question))
        return {doc_id: float(score) for doc_id, score in zip(self.ids, scores)}

    def ranked_from_scores(self, scores: Dict[str, float], final_top_k: int) -> List[Dict[str, Any]]:
        ranked_ids = sorted(scores, key=lambda doc_id: scores[doc_id], reverse=True)[:final_top_k]
        retrieved = []
        for rank, doc_id in enumerate(ranked_ids, start=1):
            record = self.id_to_record[doc_id]
            retrieved.append(
                {
                    "rank": rank,
                    "document_id": doc_id,
                    "distance": None,
                    "metadata": record["metadata"],
                    "text_preview": record["document"][:500].replace("\n", " "),
                }
            )
        return retrieved

    def retrieve(
        self,
        question: str,
        mode: str,
        final_top_k: int,
        hybrid_weight_dense: float | None,
        apply_type_filter: bool = True,
    ) -> List[Dict[str, Any]]:
        if mode == "dense":
            ranked = self.ranked_from_scores(self.dense_scores(question), final_top_k)
            return apply_document_type_ratio(ranked, self.args.document_type_ratio) if apply_type_filter else ranked
        if mode == "bm25":
            ranked = self.ranked_from_scores(self.bm25_scores(question), final_top_k)
            return apply_document_type_ratio(ranked, self.args.document_type_ratio) if apply_type_filter else ranked
        if mode == "hybrid":
            if hybrid_weight_dense is None:
                raise ValueError("hybrid_weight_dense is required for hybrid mode")
            dense = normalize_scores(self.dense_scores(question))
            bm25 = normalize_scores(self.bm25_scores(question))
            all_ids = set(dense) | set(bm25)
            scores = {
                doc_id: hybrid_weight_dense * dense.get(doc_id, 0.0)
                + (1.0 - hybrid_weight_dense) * bm25.get(doc_id, 0.0)
                for doc_id in all_ids
            }
            ranked = self.ranked_from_scores(scores, final_top_k)
            return apply_document_type_ratio(ranked, self.args.document_type_ratio) if apply_type_filter else ranked
        raise ValueError(f"Unsupported retriever mode: {mode}")


def combo_records(final_top_k_values: List[int]) -> List[Dict[str, Any]]:
    combos: List[Dict[str, Any]] = []
    for top_k in final_top_k_values:
        combos.append(
            {
                "name": f"dense_top{top_k}",
                "retriever_mode": "dense",
                "final_top_k": top_k,
                "hybrid_weight_dense": None,
            }
        )
    for top_k in final_top_k_values:
        combos.append(
            {
                "name": f"bm25_top{top_k}",
                "retriever_mode": "bm25",
                "final_top_k": top_k,
                "hybrid_weight_dense": None,
            }
        )
    for weight in HYBRID_WEIGHTS:
        weight_name = f"{int(weight * 100):03d}"
        for top_k in final_top_k_values:
            combos.append(
                {
                    "name": f"hybrid_w{weight_name}_top{top_k}",
                    "retriever_mode": "hybrid",
                    "final_top_k": top_k,
                    "hybrid_weight_dense": weight,
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
        "group": "A_no_reranker",
        "combo_count": len(combos),
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


def write_ranked_csv(path: Path, combo_results: List[Dict[str, Any]]) -> None:
    rows = []
    for result in combo_results:
        summary = result["summary"]
        combo = result["combo"]
        rows.append(
            {
                "combo_name": combo["name"],
                "retriever_mode": combo["retriever_mode"],
                "candidate_top_k": combo["final_top_k"],
                "final_top_k": result["filtered_result_k"],
                "hybrid_weight_dense": "" if combo["hybrid_weight_dense"] is None else combo["hybrid_weight_dense"],
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
        "retriever_mode",
        "candidate_top_k",
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
    combos = combo_records(args.final_top_k_values)
    if args.document_type_ratio.enabled:
        too_small = [combo["final_top_k"] for combo in combos if combo["final_top_k"] < args.document_type_ratio.final_k]
        if too_small:
            raise ValueError(
                "When RETRIEVER_DOCUMENT_TYPE_RATIO is enabled, every filter candidate top-k "
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
    for index, combo in enumerate(combos, start=1):
        print(f"[{index}/{len(combos)}] Running {combo['name']}")
        combo_dir = run_dir / combo["name"]
        output_csv = combo_dir / "retrieval_eval.csv"
        summary_json = combo_dir / "retrieval_eval_summary.json"
        started = time.perf_counter()
        rows = []
        filtered_result_k = args.document_type_ratio.final_k if args.document_type_ratio.enabled else combo["final_top_k"]
        for qa in qa_rows:
            query_started = time.perf_counter()
            retrieved = retriever.retrieve(
                qa["question"],
                combo["retriever_mode"],
                combo["final_top_k"],
                combo["hybrid_weight_dense"],
            )
            latency_ms = (time.perf_counter() - query_started) * 1000
            score = score_retrieved(qa, retrieved, args.overlap_threshold)
            row_args = argparse.Namespace(top_k=filtered_result_k, overlap_threshold=args.overlap_threshold)
            rows.append(build_result_row(qa, retrieved, score, latency_ms, row_args))

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
                "retriever_mode": combo["retriever_mode"],
                "candidate_top_k": combo["final_top_k"],
                "final_top_k": filtered_result_k,
                "document_type_ratio_filter": ratio_manifest(args.document_type_ratio),
                "hybrid_weight_dense": combo["hybrid_weight_dense"],
            }
        )
        combo_ers = ers(summary)
        combo_rers = rers(summary, filtered_result_k)
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
        print(
            "  Completed "
            f"ERS={combo_ers:.4f} RERS={combo_rers:.4f} "
            f"path@k={summary['path_hit_at_k']:.4f} "
            f"anchor@k={summary['anchor_hit_at_k']:.4f}"
        )

    write_ranked_csv(run_dir / "retriever_group_a_rers_ranked.csv", combo_results)
    print(f"Run manifest: {run_dir / 'run_manifest.json'}")
    print(f"Per-combo results: {results_path}")
    print(f"Ranked CSV: {run_dir / 'retriever_group_a_rers_ranked.csv'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run retriever-layer Group A no-reranker matrix.")
    parser.add_argument("--qa", default="benchmarking/qa_sets/retrieval_qa_paired_candidates.csv")
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--embedding-model", default=DEFAULT_MODEL)
    parser.add_argument("--document-prefix", default=DEFAULT_DOC_PREFIX)
    parser.add_argument("--query-prefix", default=DEFAULT_QUERY_PREFIX)
    parser.add_argument("--overlap-threshold", type=float, default=0.2)
    parser.add_argument("--output-root", default="benchmarking/retriever_runs")
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--final-top-k-values", type=int, nargs="+", default=FINAL_TOP_K_VALUES)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--append", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
