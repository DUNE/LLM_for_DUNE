
#!/usr/bin/env python3
"""Propose quality decisions for generated benchmark questions."""

import argparse
import json
from pathlib import Path

import requests

from config import FERMILAB_BASE_URL
from benchmarking.parent_child_comparison.generate_questions import (
    MODEL,
    parse_json_response,
)

BASE = Path("benchmarking/parent_child_comparison/data")

SYSTEM_PROMPT = """
You are reviewing candidate questions for a DUNE documentation
retrieval benchmark.

Assess each question using ONLY the supplied question, reference
answer, evidence quotation, and source title.

Evaluate:
1. Evidence: Does the quotation support the reference answer?
2. Clarity: Is the question independently understandable?
3. Relevance: Does it test meaningful DUNE-related scientific,
   technical, engineering, operational, or procedural information?
4. Answerability: Is the answer specific and unambiguous?

Reject:
- Arbitrary numeric-position or table-cell lookups.
- Page numbers, timestamps, and trivial metadata lookups.
- Questions referring vaguely to "the passage" or "the text".
- Questions whose answers are unsupported or ambiguous.
- Tautological questions with uninformative answers.

Do not reject solely because a document is old or obsolete.
Judge whether the question is correctly framed as a historical fact.

If the supplied quotation is insufficient to verify the answer,
return "needs_review".

Return ONLY a JSON object:
{
  "suggested_status": "accepted|rejected|needs_review",
  "reason": "Brief explanation"
}
"""


def load_jsonl(path):
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def evaluate(record, timeout):
    content = {
        key: record.get(key, "")
        for key in (
            "source",
            "source_title",
            "question",
            "reference_answer",
            "evidence_quote",
        )
    }

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(content, ensure_ascii=False),
            },
        ],
        "stream": False,
        "temperature": 0,
    }

    response = requests.post(
        FERMILAB_BASE_URL,
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()

    message = response.json()["choices"][0]["message"]["content"]
    result = parse_json_response(message)

    status = result.get("suggested_status")
    reason = result.get("reason")

    if status not in ("accepted", "rejected", "needs_review"):
        raise ValueError(f"Invalid status: {status}")

    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Missing review reason")

    return {
        "question_id": record["question_id"],
        "suggested_status": status,
        "reason": reason.strip(),
        "review_model": MODEL,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=BASE / "questions_reviewed.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE / "quality_suggestions.jsonl",
    )
    parser.add_argument("--max-new", type=int, default=15)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    if args.max_new < 1:
        parser.error("--max-new must be positive")
    if args.input.resolve() == args.output.resolve():
        parser.error("Input and output must be different files")

    records = load_jsonl(args.input)
    previous = load_jsonl(args.output)
    completed = {r["question_id"] for r in previous}

    count = 0
    failures = 0

    with args.output.open("a", encoding="utf-8") as f:
        for record in records:
            if record["question_id"] in completed:
                continue
            if count >= args.max_new:
                break

            print(f"Reviewing {record['question_id']} ...", flush=True)

            try:
                result = evaluate(record, args.timeout)
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
                completed.add(record["question_id"])
                count += 1
                print(
                    f"  {result['suggested_status']}: "
                    f"{result['reason']}",
                    flush=True,
                )
            except Exception as exc:
                failures += 1
                print(f"  FAILED: {type(exc).__name__}: {exc}", flush=True)
                # Avoid an infinite loop if the service is unavailable.
                break

    print("\nNew reviews:", count)
    print("Failures:", failures)
    print("Total suggestions:", len(completed))


if __name__ == "__main__":
    main()

