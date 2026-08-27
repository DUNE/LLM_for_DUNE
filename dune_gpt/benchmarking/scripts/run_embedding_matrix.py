#!/usr/bin/env python3
"""
Run a matrix of local indexing + retrieval-evaluation benchmark combos.

This runner is intended for preflight and batch benchmarking. It keeps a
manifest of each combo, records stdout/stderr logs, and can skip combos that
already completed successfully.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/multi-qa-mpnet-base-dot-v1",
    "BAAI/bge-base-en-v1.5",
]
DEFAULT_CHUNK_SIZES = [1000, 2000]
DEFAULT_CHUNK_OVERLAPS = [0, 100, 200]


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def slugify_model(model: str) -> str:
    name = model.replace("sentence-transformers/", "")
    name = name.replace("/", "_")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def combo_name(model: str, chunk_size: int, chunk_overlap: int, chunk_strategy: str, source: str, limit: int | None) -> str:
    parts = [
        slugify_model(model),
        chunk_strategy,
        f"chunk{chunk_size}",
        f"overlap{chunk_overlap}",
        source,
    ]
    if limit is not None:
        parts.append(f"limit{limit}")
    return "_".join(parts)


def parse_csv_ints(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_strings(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def default_manifest(run_dir: Path, args: argparse.Namespace, combos: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable,
        "run_dir": str(run_dir),
        "qa_path": str(resolve_path(args.qa)),
        "source": args.source,
        "limit": args.limit,
        "qa_limit": args.qa_limit,
        "top_k": args.top_k,
        "chunk_strategy": args.chunk_strategy,
        "document_prefix": args.document_prefix,
        "query_prefix": args.query_prefix,
        "combo_count": len(combos),
        "combos": combos,
    }


def run_command(command: List[str], stdout_path: Path, stderr_path: Path) -> subprocess.CompletedProcess[str]:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open("w", encoding="utf-8") as stderr_file:
        return subprocess.run(
            command,
            cwd=str(ROOT),
            text=True,
            stdout=stdout_file,
            stderr=stderr_file,
            check=False,
        )


def combo_status_path(combo_dir: Path) -> Path:
    return combo_dir / "status.json"


def combo_completed(combo_dir: Path) -> bool:
    status_path = combo_status_path(combo_dir)
    if not status_path.exists():
        return False
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return status.get("status") == "success"


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def append_jsonl(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(data, ensure_ascii=False) + "\n")


def build_combo_records(args: argparse.Namespace) -> List[Dict[str, Any]]:
    combos = []
    for model, chunk_size, chunk_overlap in itertools.product(args.models, args.chunk_sizes, args.chunk_overlaps):
        combos.append(
            {
                "embedding_model": model,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "chunk_strategy": args.chunk_strategy,
                "source": args.source,
                "limit": args.limit,
                "qa_limit": args.qa_limit,
                "top_k": args.top_k,
                "name": combo_name(model, chunk_size, chunk_overlap, args.chunk_strategy, args.source, args.limit),
            }
        )
    return combos


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a benchmark matrix for embedding/chunking combos.")
    parser.add_argument("--benchmark-config", default="benchmarking/local_index_config.example.json")
    parser.add_argument("--qa", default="benchmarking/qa_sets/retrieval_qa.csv")
    parser.add_argument("--source", choices=["docdb", "indico", "both"], default="both")
    parser.add_argument("--limit", type=int, default=None, help="Attachment limit for local indexing.")
    parser.add_argument("--qa-limit", type=int, default=None, help="QA limit for evaluation.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--chunk-strategy", choices=["word", "char"], default="word")
    parser.add_argument("--models", type=parse_csv_strings, default=DEFAULT_MODELS)
    parser.add_argument("--chunk-sizes", type=parse_csv_ints, default=DEFAULT_CHUNK_SIZES)
    parser.add_argument("--chunk-overlaps", type=parse_csv_ints, default=DEFAULT_CHUNK_OVERLAPS)
    parser.add_argument("--run-name", default=None, help="Optional run directory name under benchmarking/matrix_runs.")
    parser.add_argument("--matrix-root", default="benchmarking/matrix_runs")
    parser.add_argument("--skip-completed", action="store_true", help="Skip combos whose status.json already reports success.")
    parser.add_argument("--dry-run", action="store_true", help="Print combos and exit without running.")
    parser.add_argument("--device", default=None, help="Optional evaluator device override, e.g. cpu or cuda.")
    parser.add_argument("--document-prefix", default="passage: ", help="Evaluator document prefix.")
    parser.add_argument("--query-prefix", default="passage: ", help="Evaluator query prefix.")
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    combos = build_combo_records(args)
    run_name = args.run_name or datetime.now().strftime("matrix_%Y%m%d_%H%M%S")
    run_dir = resolve_path(args.matrix_root) / run_name
    manifest_path = run_dir / "run_manifest.json"
    results_path = run_dir / "combo_results.jsonl"

    if args.dry_run:
        print(json.dumps(default_manifest(run_dir, args, combos), indent=2, ensure_ascii=False))
        return

    write_json(manifest_path, default_manifest(run_dir, args, combos))

    for index, combo in enumerate(combos, start=1):
        name = combo["name"]
        combo_dir = run_dir / name
        index_dir = resolve_path("benchmarking/chroma_experiments") / name
        result_csv = combo_dir / "retrieval_eval.csv"
        result_summary = combo_dir / "retrieval_eval_summary.json"
        status_path = combo_status_path(combo_dir)

        if args.skip_completed and combo_completed(combo_dir):
            print(f"[{index}/{len(combos)}] Skipping completed combo: {name}")
            append_jsonl(
                results_path,
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "combo": combo,
                    "status": "skipped_completed",
                    "combo_dir": str(combo_dir),
                    "index_dir": str(index_dir),
                },
            )
            continue

        print(f"[{index}/{len(combos)}] Running combo: {name}")
        combo_started = time.perf_counter()
        write_json(
            status_path,
            {
                "status": "running",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "combo": combo,
                "combo_dir": str(combo_dir),
                "index_dir": str(index_dir),
            },
        )

        index_command = [
            sys.executable,
            "cli.py",
            "index-local",
            "--benchmark-config",
            args.benchmark_config,
            "--source",
            args.source,
            "--data-path",
            str(index_dir),
            "--chunk-size",
            str(combo["chunk_size"]),
            "--chunk-overlap",
            str(combo["chunk_overlap"]),
            "--chunk-strategy",
            combo["chunk_strategy"],
            "--embedding-model",
            combo["embedding_model"],
        ]
        if args.limit is not None:
            index_command.extend(["--limit", str(args.limit)])

        index_stdout = combo_dir / "index_stdout.log"
        index_stderr = combo_dir / "index_stderr.log"
        index_started = time.perf_counter()
        index_result = run_command(index_command, index_stdout, index_stderr)
        index_elapsed = time.perf_counter() - index_started
        if index_result.returncode != 0:
            failure = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "combo": combo,
                "status": "index_failed",
                "combo_dir": str(combo_dir),
                "index_dir": str(index_dir),
                "index_returncode": index_result.returncode,
                "index_elapsed_seconds": round(index_elapsed, 3),
                "index_stdout": str(index_stdout),
                "index_stderr": str(index_stderr),
            }
            write_json(status_path, failure)
            append_jsonl(results_path, failure)
            print(f"  Indexing failed for {name}; see {index_stderr}")
            continue

        eval_command = [
            sys.executable,
            str(Path("benchmarking") / "scripts" / "evaluate_retrieval.py"),
            "--qa",
            args.qa,
            "--data-path",
            str(index_dir),
            "--embedding-model",
            combo["embedding_model"],
            "--top-k",
            str(args.top_k),
            "--output",
            str(result_csv),
            "--summary-output",
            str(result_summary),
            "--document-prefix",
            args.document_prefix,
            "--query-prefix",
            args.query_prefix,
            "--progress-every",
            str(args.progress_every),
        ]
        if args.qa_limit is not None:
            eval_command.extend(["--limit", str(args.qa_limit)])
        if args.device:
            eval_command.extend(["--device", args.device])

        eval_stdout = combo_dir / "eval_stdout.log"
        eval_stderr = combo_dir / "eval_stderr.log"
        eval_started = time.perf_counter()
        eval_result = run_command(eval_command, eval_stdout, eval_stderr)
        eval_elapsed = time.perf_counter() - eval_started
        if eval_result.returncode != 0:
            failure = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "combo": combo,
                "status": "eval_failed",
                "combo_dir": str(combo_dir),
                "index_dir": str(index_dir),
                "index_elapsed_seconds": round(index_elapsed, 3),
                "eval_returncode": eval_result.returncode,
                "eval_elapsed_seconds": round(eval_elapsed, 3),
                "index_stdout": str(index_stdout),
                "index_stderr": str(index_stderr),
                "eval_stdout": str(eval_stdout),
                "eval_stderr": str(eval_stderr),
            }
            write_json(status_path, failure)
            append_jsonl(results_path, failure)
            print(f"  Evaluation failed for {name}; see {eval_stderr}")
            continue

        summary = {}
        if result_summary.exists():
            summary = json.loads(result_summary.read_text(encoding="utf-8"))

        success = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "combo": combo,
            "status": "success",
            "combo_dir": str(combo_dir),
            "index_dir": str(index_dir),
            "index_elapsed_seconds": round(index_elapsed, 3),
            "eval_elapsed_seconds": round(eval_elapsed, 3),
            "total_elapsed_seconds": round(time.perf_counter() - combo_started, 3),
            "result_csv": str(result_csv),
            "result_summary": str(result_summary),
            "index_stdout": str(index_stdout),
            "index_stderr": str(index_stderr),
            "eval_stdout": str(eval_stdout),
            "eval_stderr": str(eval_stderr),
            "metrics": {
                "path_hit_at_k": summary.get("path_hit_at_k"),
                "anchor_hit_at_k": summary.get("anchor_hit_at_k"),
                "mrr_path": summary.get("mrr_path"),
                "mrr_anchor": summary.get("mrr_anchor"),
                "mean_best_anchor_overlap": summary.get("mean_best_anchor_overlap"),
                "mean_latency_ms": summary.get("mean_latency_ms"),
            },
        }
        write_json(status_path, success)
        append_jsonl(results_path, success)
        print(
            "  Completed "
            f"path_hit@{args.top_k}={summary.get('path_hit_at_k')} "
            f"anchor_hit@{args.top_k}={summary.get('anchor_hit_at_k')}"
        )

    print(f"Run manifest: {manifest_path}")
    print(f"Per-combo results: {results_path}")


if __name__ == "__main__":
    main()
