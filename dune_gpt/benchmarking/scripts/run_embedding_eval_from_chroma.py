#!/usr/bin/env python3
"""Evaluate the 24 embedding-layer Chroma indexes without rebuilding them."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from retrieval_scores import ERS_WEIGHTS, compute_ers


ROOT = Path(__file__).resolve().parents[2]

COMBOS = [
    {
        "model": "allMiniLM",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "slug": "all-MiniLM-L6-v2",
        "doc_prefix": "passage: ",
        "query_prefix": "passage: ",
    },
    {
        "model": "MPNet",
        "embedding_model": "sentence-transformers/multi-qa-mpnet-base-dot-v1",
        "slug": "multi-qa-mpnet-base-dot-v1",
        "doc_prefix": "passage: ",
        "query_prefix": "passage: ",
    },
    {
        "model": "BGE",
        "embedding_model": "BAAI/bge-base-en-v1.5",
        "slug": "BAAI_bge-base-en-v1.5",
        "doc_prefix": "passage: ",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
    },
    {
        "model": "e5-small-v2",
        "embedding_model": "intfloat/e5-small-v2",
        "slug": "intfloat_e5-small-v2",
        "doc_prefix": "passage: ",
        "query_prefix": "query: ",
    },
]

CHUNK_SIZES = [1000, 2000]
CHUNK_OVERLAPS = [0, 100, 200]
def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def combo_name(slug: str, chunk_size: int, chunk_overlap: int) -> str:
    return f"{slug}_word_chunk{chunk_size}_overlap{chunk_overlap}_both"


def run_command(command: list[str], stdout_path: Path, stderr_path: Path) -> int:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open("w", encoding="utf-8") as stderr_file:
        completed = subprocess.run(
            command,
            cwd=str(ROOT),
            text=True,
            stdout=stdout_file,
            stderr=stderr_file,
            check=False,
            env={**os.environ, "TOKENIZERS_PARALLELISM": "false"},
        )
    return completed.returncode


def load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run embedding evaluation against existing Chroma indexes.")
    parser.add_argument("--qa", default="benchmarking/qa_sets/retrieval_qa_paired_candidates.csv")
    parser.add_argument("--chroma-root", default="benchmarking/chroma_experiments")
    parser.add_argument("--run-dir", default="benchmarking/matrix_runs/local_embedding_24combo_paired_qa_eval")
    parser.add_argument("--ranked-output", default="benchmarking/results/embedding_layer_24combo_paired_qa_ers.csv")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--overlap-threshold", type=float, default=0.2)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-completed", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    qa_path = resolve(args.qa)
    chroma_root = resolve(args.chroma_root)
    run_dir = resolve(args.run_dir)
    ranked_output = resolve(args.ranked_output)
    run_dir.mkdir(parents=True, exist_ok=True)
    ranked_output.parent.mkdir(parents=True, exist_ok=True)

    results_jsonl = run_dir / "combo_results.jsonl"
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qa": str(qa_path),
        "chroma_root": str(chroma_root),
        "top_k": args.top_k,
        "overlap_threshold": args.overlap_threshold,
        "ers_weights": ERS_WEIGHTS,
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    ranked_rows: list[dict[str, Any]] = []
    total = len(COMBOS) * len(CHUNK_SIZES) * len(CHUNK_OVERLAPS)
    completed_count = 0

    for combo in COMBOS:
        for chunk_size in CHUNK_SIZES:
            for chunk_overlap in CHUNK_OVERLAPS:
                completed_count += 1
                name = combo_name(combo["slug"], chunk_size, chunk_overlap)
                data_path = chroma_root / name
                combo_dir = run_dir / name
                result_csv = combo_dir / "retrieval_eval.csv"
                result_summary = combo_dir / "retrieval_eval_summary.json"
                stdout_path = combo_dir / "eval_stdout.log"
                stderr_path = combo_dir / "eval_stderr.log"

                if not (data_path / "chroma.sqlite3").exists():
                    raise RuntimeError(f"Missing Chroma sqlite file: {data_path / 'chroma.sqlite3'}")

                print(f"[{completed_count}/{total}] Evaluating {name}", flush=True)
                started = time.perf_counter()
                status = "success"
                returncode = 0

                if not (args.skip_completed and result_summary.exists()):
                    command = [
                        sys.executable,
                        str(Path("benchmarking") / "scripts" / "evaluate_retrieval.py"),
                        "--qa",
                        str(qa_path),
                        "--data-path",
                        str(data_path),
                        "--embedding-model",
                        combo["embedding_model"],
                        "--top-k",
                        str(args.top_k),
                        "--overlap-threshold",
                        str(args.overlap_threshold),
                        "--output",
                        str(result_csv),
                        "--summary-output",
                        str(result_summary),
                        "--document-prefix",
                        combo["doc_prefix"],
                        "--query-prefix",
                        combo["query_prefix"],
                        "--progress-every",
                        str(args.progress_every),
                    ]
                    if args.device:
                        command.extend(["--device", args.device])
                    returncode = run_command(command, stdout_path, stderr_path)
                    if returncode != 0:
                        status = "eval_failed"

                elapsed = time.perf_counter() - started
                record: dict[str, Any] = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "combo": {
                        "embedding_model": combo["embedding_model"],
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                        "chunk_strategy": "word",
                        "source": "both",
                        "top_k": args.top_k,
                        "name": name,
                    },
                    "status": status,
                    "combo_dir": str(combo_dir),
                    "index_dir": str(data_path),
                    "eval_returncode": returncode,
                    "eval_elapsed_seconds": round(elapsed, 3),
                    "result_csv": str(result_csv),
                    "result_summary": str(result_summary),
                    "eval_stdout": str(stdout_path),
                    "eval_stderr": str(stderr_path),
                }

                if status != "success":
                    with results_jsonl.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    raise RuntimeError(f"Evaluation failed for {name}; see {stderr_path}")

                summary = load_summary(result_summary)
                metrics = {
                    "path_at_k": summary["path_hit_at_k"],
                    "anchor_at_k": summary["anchor_hit_at_k"],
                    "mrr_path": summary["mrr_path"],
                    "mrr_anchor": summary["mrr_anchor"],
                    "best_ov": summary["mean_best_anchor_overlap"],
                    "lat_ms": summary["mean_latency_ms"],
                }
                record["metrics"] = metrics
                with results_jsonl.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

                ranked_row = {
                    "model": combo["model"],
                    "embedding_model": combo["embedding_model"],
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "chunk_strategy": "word",
                    "source": "both",
                    "top_k": args.top_k,
                    "doc_prefix": combo["doc_prefix"],
                    "query_prefix": combo["query_prefix"],
                    **metrics,
                    "combo_name": name,
                    "run_source": str(result_summary),
                }
                ranked_row["ERS"] = compute_ers(ranked_row)
                ranked_rows.append(ranked_row)
                print(
                    "  "
                    f"path@{args.top_k}={metrics['path_at_k']:.4f} "
                    f"anchor@{args.top_k}={metrics['anchor_at_k']:.4f} "
                    f"ERS={ranked_row['ERS']:.4f}",
                    flush=True,
                )

    ranked_rows.sort(key=lambda item: item["ERS"], reverse=True)
    for rank, row in enumerate(ranked_rows, start=1):
        row["rank"] = rank

    fieldnames = [
        "rank",
        "model",
        "embedding_model",
        "chunk_size",
        "chunk_overlap",
        "chunk_strategy",
        "source",
        "top_k",
        "doc_prefix",
        "query_prefix",
        "path_at_k",
        "anchor_at_k",
        "mrr_path",
        "mrr_anchor",
        "best_ov",
        "lat_ms",
        "ERS",
        "combo_name",
        "run_source",
    ]
    with ranked_output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ranked_rows)

    print(f"Wrote ranked results: {ranked_output}", flush=True)


if __name__ == "__main__":
    main()
