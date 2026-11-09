import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from src.retrieval.unified_backend import UnifiedRetrievalBackend
from src.api.fermilab_client import FermilabAPIClient

BASE = Path("benchmarking/parent_child_comparison")
QUESTIONS = BASE / "data/questions_reviewed.jsonl"
OUTPUT = BASE / "results/parent_child_generation_full.jsonl"

INDEX = (
    "/exp/dune/app/users/dune/askdune/dev/"
    "docdb-integration/test_data/benchmark_200"
)

MODEL = "gpt-oss:120b"
TOP_K = 5
CANDIDATE_COUNT = 30


def load_questions():
    with QUESTIONS.open() as f:
        return [
            q
            for line in f
            if (q := json.loads(line))["quality_status"] == "accepted"
        ]


def completed_ids():
    if not OUTPUT.exists():
        return set()

    completed = set()

    with OUTPUT.open() as f:
        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)

            if record.get("status") == "success":
                completed.add(record["question_id"])

    return completed


def generate_answer(client, question, context, references):
    chunks = []

    for chunk in client.chat_completion(
        question=question,
        context=context,
        model=MODEL,
        links=references,
    ):
        if not isinstance(chunk, str):
            continue

        if chunk.startswith("STATUS:"):
            continue

        if chunk.startswith("SOURCES:"):
            continue

        chunks.append(chunk)

    return "".join(chunks).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    questions = load_questions()
    completed = completed_ids()

    pending = [
        q for q in questions
        if q["question_id"] not in completed
    ]

    if args.limit is not None:
        pending = pending[:args.limit]

    print(f"Accepted questions: {len(questions)}", flush=True)
    print(f"Already completed: {len(completed)}", flush=True)
    print(f"Questions to process: {len(pending)}", flush=True)

    if not pending:
        return

    backend = UnifiedRetrievalBackend(INDEX)
    client = FermilabAPIClient()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    for index, q in enumerate(pending, 1):
        qid = q["question_id"]

        print(
            f"\n[{index}/{len(pending)}] {qid}",
            flush=True,
        )

        record = {
            "question_id": qid,
            "question": q["question"],
            "reference_answer": q["reference_answer"],
            "evidence_quote": q["evidence_quote"],
            "expected_source": q["source"],
            "expected_source_id": q["source_id"],
            "expected_parent_id": q["parent_id"],
            "expected_source_url": q["source_url"],
            "model": MODEL,
            "candidate_count": CANDIDATE_COUNT,
            "top_k": TOP_K,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "status": "error",
        }

        try:
            start = time.perf_counter()

            contexts, references = backend.search(
                q["question"],
                top_k=TOP_K,
                k_embeddings=CANDIDATE_COUNT,
            )

            retrieval_seconds = time.perf_counter() - start

            context_text = "\n\n".join(contexts)

            start = time.perf_counter()

            answer = generate_answer(
                client,
                q["question"],
                context_text,
                references,
            )

            generation_seconds = time.perf_counter() - start

            if not answer or answer.startswith("[ERROR]"):
                raise RuntimeError(
                    f"Generation failed: {answer[:300]}"
                )

            record.update({
                "status": "success",
                "generated_answer": answer,
                "contexts": contexts,
                "references": references,
                "retrieved_contexts": len(contexts),
                "context_words": len(context_text.split()),
                "retrieval_seconds": round(retrieval_seconds, 3),
                "generation_seconds": round(generation_seconds, 3),
            })

            print(
                f"SUCCESS | "
                f"retrieval={retrieval_seconds:.1f}s | "
                f"generation={generation_seconds:.1f}s | "
                f"contexts={len(contexts)}",
                flush=True,
            )

        except Exception as exc:
            record["error"] = str(exc)
            print(f"ERROR: {exc}", flush=True)

        with OUTPUT.open("a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

    print("\nBenchmark run finished.", flush=True)


if __name__ == "__main__":
    main()
