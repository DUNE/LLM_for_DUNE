
#!/usr/bin/env python3
"""Generate 75 DocDB + 75 Indico grounded benchmark questions."""

import argparse
import json
import random
import re
import time
import html
from pathlib import Path

import chromadb
import requests

from config import FERMILAB_BASE_URL


BASE = Path("benchmarking/parent_child_comparison")
DEFAULT_DB = Path(
    "/exp/dune/app/users/dune/askdune/dev/"
    "docdb-integration/test_data/benchmark_200"
)

MODEL = "gpt-oss:120b"
SEED = 2026
QUESTIONS_PER_SOURCE = 75



SYSTEM_PROMPT = """
You are creating a rigorous retrieval benchmark for DUNE documentation.

Generate exactly ONE specific, answerable question using ONLY the
provided source passage.

Requirements:
- The question must be answerable from the supplied passage.
- Prefer meaningful scientific, technical, procedural, or numerical facts.
- Every question must be understandable independently.
- If a question refers to a figure, table, slide, section, specific result,
  or statement within a document, identify the relevant document or
  presentation by its title in the question.
- Do not ask "What does Figure 3 show?" without naming the source.
- Do not ask questions requiring visual interpretation of figures.
- Figure-related questions are permitted only when the extracted text
  explicitly states the answer.
- Do not refer vaguely to "this document", "the presentation",
  "the passage", or "the text".
- Do not invent information.
- The reference answer must be concise and supported by the passage.
- Include a short VERBATIM evidence quotation copied from the passage.
- The quotation must contain the information needed to answer.
- Return ONLY a JSON object with exactly these keys:
  "question", "reference_answer", "evidence_quote"
"""


def parse_json_response(text):
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")

        if start < 0 or end <= start:
            raise ValueError("No JSON object found")

        return json.loads(text[start:end + 1])


def normalize_passage(text):
    """Normalize formatting while preserving source wording."""
    text = html.unescape(text)
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"</?[^>]+>", " ", text)
    text = re.sub(r"\*\*|__", "", text)
    text = re.sub(r"(?<!\w)[*_](?=\w)|(?<=\w)[*_](?!\w)", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def generate_question(passage, source, source_id, title, timeout):
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Source type: {source}\n"
                    f"Source ID: {source_id}\n"
                    f"Document/presentation title: {title}\n\n"
                    f"PASSAGE:\n{normalize_passage(passage)}"
                ),
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

    result = response.json()
    message = result["choices"][0]["message"]["content"]

    if not isinstance(message, str):
        raise ValueError("Model returned no text")

    record = parse_json_response(message)

    for key in ("question", "reference_answer", "evidence_quote"):
        if not isinstance(record.get(key), str):
            raise ValueError(f"Missing or invalid field: {key}")

        record[key] = record[key].strip()

        if not record[key]:
            raise ValueError(f"Empty field: {key}")


    quote = record["evidence_quote"]

    # Remove quotation marks surrounding the evidence.
    quote = quote.strip().strip('"“”').strip("'‘’").strip()
    record["evidence_quote"] = quote

    # Normalize formatting in both the evidence and indexed passage.
    normalized_quote = normalize_passage(quote)
    normalized_passage = normalize_passage(passage)

    if not normalized_quote or normalized_quote not in normalized_passage:
        raise ValueError(
            "Evidence quotation does not match the normalized indexed passage"
        )

    if len(record["question"]) < 20:
        raise ValueError("Question is unusually short")

    return record

def load_jsonl(path):
    records = []

    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))

    return records


def append_jsonl(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def select_sources(manifest):
    rng = random.Random(SEED)
    original_groups = {}

    for source in ("docdb", "indico"):
        candidates = [
            item for item in manifest["sources"]
            if item["source"] == source
            and item["child_count"] > 0
        ]

        if len(candidates) < QUESTIONS_PER_SOURCE:
            raise RuntimeError(
                f"Only {len(candidates)} eligible {source} sources"
            )

        original_groups[source] = (
            candidates,
            rng.sample(candidates, 50)
        )

    selected = []

    for source in ("docdb", "indico"):
        candidates, original = original_groups[source]

        original_keys = {
            str(item["source_id"]) for item in original
        }

        remaining = [
            item for item in candidates
            if str(item["source_id"]) not in original_keys
        ]

        additional = rng.sample(
            remaining,
            QUESTIONS_PER_SOURCE - 50
        )

        selected.extend(original + additional)

    return selected


def choose_passage(collection, source_entry):
    source = source_entry["source"]
    source_id = source_entry["source_id"]

    id_field = "docdb_id" if source == "docdb" else "indico_id"

    result = collection.get(
        where={id_field: source_id},
        include=["documents", "metadatas"],
    )

    candidates = []

    for child_id, text, meta in zip(
        result["ids"],
        result["documents"],
        result["metadatas"],
    ):
        meta = meta or {}
        text = (text or "").strip()

        if source == "docdb":
            if str(meta.get("docdb_version")) != str(
                source_entry["active_version"]
            ):
                continue

        if source == "indico":
            if meta.get("index_status") != "complete":
                continue

        if len(text) < 250:
            continue

        candidates.append({
            "child_id": child_id,
            "parent_id": str(meta.get("parent_id", "")),
            "child_index": meta.get("child_index"),
            "text": text,
            "metadata": meta,
        })

    if not candidates:
        raise ValueError("No sufficiently informative child chunks")

    # Deterministic sampling, independent of ChromaDB return order.
    candidates.sort(key=lambda item: item["child_id"])

    rng = random.Random(f"{SEED}:{source}:{source_id}")

    # Prefer later portions of a source when possible.
    later = [
        item for item in candidates
        if isinstance(item["child_index"], int)
        and item["child_index"] > 0
    ]

    pool = later if later and rng.random() < 0.6 else candidates

    return rng.choice(pool)


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--index-path",
        type=Path,
        default=DEFAULT_DB,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=BASE / "data/corpus_manifest.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE / "data/questions_generated.jsonl",
    )
    parser.add_argument(
        "--failures",
        type=Path,
        default=BASE / "data/question_failures.jsonl",
    )
    parser.add_argument(
        "--max-new",
        type=int,
        default=100,
        help="Maximum new questions to generate in this run",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
    )
    parser.add_argument(
        "--skip-failed",
        action="store_true",
        help="Skip sources recorded in the failure log",
    )

    args = parser.parse_args()

    db_path = args.index_path.resolve()

    if db_path.name != "benchmark_200" or not db_path.is_dir():
        parser.error("Only the existing benchmark_200 database is allowed")

    if args.max_new < 1:
        parser.error("--max-new must be positive")

    manifest = json.loads(
        args.manifest.read_text(encoding="utf-8")
    )

    if manifest["source_counts"].get("docdb") != 108:
        raise RuntimeError("Unexpected DocDB corpus size")

    if manifest["source_counts"].get("indico") != 106:
        raise RuntimeError("Unexpected Indico corpus size")

    selected = select_sources(manifest)

    existing = load_jsonl(args.output)

    failed_keys = set()

    if args.skip_failed:
        for item in load_jsonl(args.failures):
            failed_keys.add(
                (str(item["source"]), str(item["source_id"]))
            )

    completed = {
        (item["source"], str(item["source_id"]))
        for item in existing
    }

    selected_keys = {
        (item["source"], str(item["source_id"]))
        for item in selected
    }

    if not completed.issubset(selected_keys):
        raise RuntimeError(
            "Output contains sources outside this fixed selection. "
            "Use a separate output file for a new experiment."
        )

    client = chromadb.PersistentClient(path=str(db_path))

    collections = {
        "docdb": client.get_collection("DUNE_DOCDB_CHILDREN"),
        "indico": client.get_collection("DUNE_INDICO_CHILDREN"),
    }

    print("Model:", MODEL, flush=True)
    print("Seed:", SEED, flush=True)
    print("Corpus sources:", len(manifest["sources"]), flush=True)
    print("Selected questions:", len(selected), flush=True)
    print("Already completed:", len(completed), flush=True)

    generated = 0
    failures = 0
    attempted = 0

    for entry in selected:
        source = entry["source"]
        source_id = str(entry["source_id"])
        key = (source, source_id)

        if key in completed:
            continue

        if args.skip_failed and key in failed_keys:
            print(
                f"Skipping previously failed {source} {source_id}",
                flush=True,
            )
            continue

        if attempted >= args.max_new:
            break

        attempted += 1

        print(
            f"\nGenerating {source} {source_id} ...",
            flush=True,
        )

        try:
            chosen = choose_passage(
                collections[source],
                entry,
            )

            passage = chosen["text"]
            answer = None
            last_error = None


            for attempt in range(1, 4):
                try:
                    answer = generate_question(
                        passage,
                        source,
                        source_id,
                        (
                            chosen["metadata"].get("presentation_title")
                            or chosen["metadata"].get("title")
                            or chosen["metadata"].get("meeting_name")
                            or ""
                        ),
                        args.timeout,
                    )
                    break

                except (
                    requests.RequestException,
                    ValueError,
                    KeyError,
                    IndexError,
                    json.JSONDecodeError,
                ) as exc:
                    last_error = exc
                    print(
                        f"  Attempt {attempt}/3 failed: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )

                    if attempt < 3:
                        time.sleep(2 * attempt)

            if answer is None:
                raise RuntimeError(
                    f"All attempts failed: {last_error}"
                )

            metadata = chosen["metadata"]

            record = {
                "question_id": (
                    f"{source}_{source_id}"
                ),
                "source": source,
                "source_id": source_id,
                "source_version": entry["active_version"],
                "parent_id": chosen["parent_id"],
                "evidence_child_id": chosen["child_id"],
                "child_index": chosen["child_index"],
                "question": answer["question"],
                "reference_answer": answer["reference_answer"],
                "evidence_quote": answer["evidence_quote"],
                "source_title": (
                    metadata.get("presentation_title")
                    or metadata.get("title")
                    or metadata.get("meeting_name")
                    or ""
                ),
                "source_url": (
                    metadata.get("attachment_url")
                    or metadata.get("event_url")
                    or metadata.get("docdb_url")
                    or ""
                ),
                "generation_model": MODEL,
                "generation_seed": SEED,
                "corpus_manifest_sha256": (
                    manifest["source_manifest_sha256"]
                ),
            }

            append_jsonl(args.output, record)

            completed.add(key)
            generated += 1

            print(
                f"  Saved ({len(completed)}/{len(selected)}): "
                f"{answer['question']}",
                flush=True,
            )

        except Exception as exc:
            failures += 1

            append_jsonl(args.failures, {
                "source": source,
                "source_id": source_id,
                "error": f"{type(exc).__name__}: {exc}",
            })

            print(
                f"  FAILED: {type(exc).__name__}: {exc}",
                flush=True,
            )

    final_records = load_jsonl(args.output)

    counts = {
        source: sum(
            item["source"] == source
            for item in final_records
        )
        for source in ("docdb", "indico")
    }

    print("\n========== QUESTION GENERATION ==========")
    print("New questions this run:", generated)
    print("Failures this run:", failures)
    print("Total saved:", len(final_records))
    print("DocDB questions:", counts["docdb"])
    print("Indico questions:", counts["indico"])
    print("Output:", args.output)
    print("=========================================")


if __name__ == "__main__":
    main()

