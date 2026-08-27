#!/usr/bin/env python3
"""
Generate paired retrieval QA rows from cached local attachments.

Each target window produces two questions sharing the same ground truth:
one lexical-anchor query and one paraphrased query. The output is a reviewable
CSV; promote reviewed rows into a formal benchmark set.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import mimetypes
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config import LITELLM_API_KEY, LITELLM_API_URL, LLM_MODEL
from src.core.local_attachment_processor import LocalTextExtractor


ALLOWED_TYPES = {"fact_lookup", "summary", "definition", "procedure", "comparison", "number_value"}
VARIANTS = ("lexical_anchor", "paraphrased")


def normalize_chat_url(url: str) -> str:
    url = (url or "").strip().rstrip("/")
    if not url:
        raise RuntimeError("LITELLM_API_URL is not configured")
    if url.endswith("/v1/chat/completions") or url.endswith("/chat/completions"):
        return url
    return f"{url}/v1/chat/completions"


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def iter_manifest(cache_path: Path, source: str) -> Iterable[Dict[str, Any]]:
    sources = ["docdb", "indico"] if source == "both" else [source]
    for source_name in sources:
        manifest = cache_path / source_name / "metadata.jsonl"
        if not manifest.exists():
            continue
        with manifest.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("error") or not record.get("path"):
                    continue
                record["_source_name"] = source_name
                yield record


def should_skip_record(record: Dict[str, Any], include_html: bool) -> bool:
    content_type = str(record.get("content_type") or "").lower()
    return bool(content_type.startswith("text/html") and not include_html)


def content_type_for(record: Dict[str, Any], path: Path) -> str:
    content_type = record.get("content_type") or ""
    if content_type:
        return content_type
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed or ""


def extract_text(record: Dict[str, Any], extractor: LocalTextExtractor) -> str:
    path = resolve_path(record["path"])
    if not path.exists():
        return ""
    content = path.read_bytes()
    raw_text, _document_type = extractor.get_raw_text(content_type_for(record, path), content)
    return re.sub(r"\s+", " ", raw_text or "").strip()


def deterministic_seed(seed_basis: str, random_seed: int) -> int:
    digest = hashlib.sha256(f"{random_seed}:{seed_basis}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def align_to_word_boundary(text: str, start: int) -> int:
    if start <= 0:
        return 0
    if start >= len(text):
        return len(text)
    while start < len(text) and not text[start].isspace() and not text[start - 1].isspace():
        start += 1
    while start < len(text) and text[start].isspace():
        start += 1
    return min(start, len(text))


def make_windows(
    text: str,
    window_chars: int,
    windows_per_attachment: int,
    seed_basis: str,
    random_seed: int,
) -> List[Dict[str, Any]]:
    if not text:
        return []
    words = text.split()

    def make_window(start: int, index: int) -> Dict[str, Any]:
        raw_window = text[start : start + window_chars]
        stripped_window = raw_window.strip()
        leading_trim = len(raw_window) - len(raw_window.lstrip())
        char_start = start + leading_trim
        char_end = char_start + len(stripped_window)
        word_start = len(text[:char_start].split())
        word_end = word_start + len(stripped_window.split())
        return {
            "window_index": index,
            "text": stripped_window,
            "anchor_start_char": char_start,
            "anchor_end_char": char_end,
            "anchor_start_word": word_start,
            "anchor_end_word": word_end,
        }

    if len(text) <= window_chars:
        return [
            {
                "window_index": 1,
                "text": text,
                "anchor_start_char": 0,
                "anchor_end_char": len(text),
                "anchor_start_word": 0,
                "anchor_end_word": len(words),
            }
        ]

    max_start = max(0, len(text) - window_chars)
    rng = random.Random(deterministic_seed(seed_basis, random_seed))
    starts = []
    seen_starts = set()
    attempts = 0
    while len(starts) < windows_per_attachment and attempts < max(50, windows_per_attachment * 10):
        attempts += 1
        start = align_to_word_boundary(text, rng.randint(0, max_start) if max_start else 0)
        if start in seen_starts:
            continue
        seen_starts.add(start)
        starts.append(start)
    if not starts:
        starts = [0]

    windows: List[Dict[str, Any]] = []
    seen = set()
    for index, start in enumerate(starts, start=1):
        window = make_window(start, index)
        if not window["text"] or window["text"] in seen:
            continue
        seen.add(window["text"])
        windows.append(window)
    return windows


def build_prompt(record: Dict[str, Any], overview: str, target_window: str) -> List[Dict[str, str]]:
    title = record.get("title") or record.get("event_title") or record.get("attachment_title") or record.get("filename", "")
    source = record.get("_source_name", "")
    return [
        {
            "role": "system",
            "content": (
                "You create paired retrieval benchmark questions for a DUNE RAG system. "
                "You will receive ATTACHMENT OVERVIEW and TARGET WINDOW. "
                "Use ATTACHMENT OVERVIEW only for orientation. "
                "Both questions and answers must be supported strictly by TARGET WINDOW. "
                "Do not use outside knowledge. Do not mention file names, URLs, local paths, document IDs, "
                "metadata, or that a target window exists. Return only valid JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Source: {source}\n"
                f"Title: {title}\n\n"
                f"ATTACHMENT OVERVIEW:\n{overview}\n\n"
                f"TARGET WINDOW:\n{target_window}\n\n"
                "Generate exactly two retrieval QA items from TARGET WINDOW:\n"
                "1. lexical_anchor: may use specific names, acronyms, metrics, detector components, "
                "processes, parameters, or numerical values from TARGET WINDOW. It should be easy to locate "
                "by exact evidence terms.\n"
                "2. paraphrased: asks for information supported by the same TARGET WINDOW but avoids copying "
                "rare exact phrases, long noun phrases, or title-like wording from TARGET WINDOW when possible. "
                "It should sound like a natural user question while preserving enough specificity to avoid ambiguity.\n\n"
                "Both items must have concise answers fully supported by TARGET WINDOW. "
                "Both items must use one of these question_type values: "
                "fact_lookup, summary, definition, procedure, comparison, number_value.\n\n"
                "Return exactly this JSON object shape:\n"
                "{\n"
                '  "lexical_anchor": {"question": "...", "answer": "...", "question_type": "..."},\n'
                '  "paraphrased": {"question": "...", "answer": "...", "question_type": "..."}\n'
                "}"
            ),
        },
    ]


def parse_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("LLM output was not a JSON object")
    return data


def call_litellm(record: Dict[str, Any], overview: str, target_window: str, args: argparse.Namespace) -> Dict[str, Dict[str, str]]:
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    payload = {
        "model": args.model,
        "messages": build_prompt(record, overview, target_window),
        "temperature": args.temperature,
        "stream": False,
    }

    last_exc: Exception | None = None
    for attempt in range(1, args.retries + 2):
        try:
            response = requests.post(args.api_url, headers=headers, json=payload, timeout=args.timeout)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            data = parse_json_object(content)
            parsed: Dict[str, Dict[str, str]] = {}
            for variant in VARIANTS:
                item = data.get(variant)
                if not isinstance(item, dict):
                    raise ValueError(f"Missing object for variant={variant}")
                question = str(item.get("question", "")).strip()
                answer = str(item.get("answer", "")).strip()
                question_type = str(item.get("question_type", "fact_lookup")).strip()
                if not question or not answer:
                    raise ValueError(f"Empty question or answer for variant={variant}")
                if question_type not in ALLOWED_TYPES:
                    question_type = "fact_lookup"
                parsed[variant] = {
                    "question": question,
                    "answer": answer,
                    "question_type": question_type,
                }
            return parsed
        except Exception as exc:
            last_exc = exc
            if attempt <= args.retries:
                time.sleep(args.retry_sleep * attempt)
    raise RuntimeError(f"LiteLLM generation failed after retries: {last_exc}")


def lexical_fallback(record: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    title = record.get("title") or record.get("event_title") or record.get("attachment_title") or record.get("filename", "this DUNE attachment")
    return {
        "lexical_anchor": {
            "question": f"What are the main points discussed in {title}?",
            "answer": "",
            "question_type": "summary",
        },
        "paraphrased": {
            "question": "What key information is presented in this material?",
            "answer": "",
            "question_type": "summary",
        },
    }


def write_rows(rows: List[Dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = csv_fieldnames()
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def csv_fieldnames() -> List[str]:
    return [
        "id",
        "window_pair_id",
        "query_variant",
        "question",
        "answer",
        "expected_local_path",
        "expected_source_url",
        "question_type",
        "anchor_window_index",
        "anchor_start_word",
        "anchor_end_word",
        "anchor_start_char",
        "anchor_end_char",
        "source",
        "attachment_title",
    ]


def initialize_output(output: Path, overwrite: bool) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        with output.open("r", newline="", encoding="utf-8") as handle:
            return max(0, sum(1 for _ in handle) - 1)

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fieldnames())
        writer.writeheader()
    return 0


def append_rows(rows: List[Dict[str, Any]], output: Path) -> None:
    with output.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fieldnames())
        writer.writerows(rows)
        handle.flush()


def completed_window_ids(output: Path) -> set[str]:
    if not output.exists():
        return set()
    with output.open("r", newline="", encoding="utf-8") as handle:
        return {row["window_pair_id"] for row in csv.DictReader(handle) if row.get("window_pair_id")}


def write_rows_legacy(rows: List[Dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "window_pair_id",
        "query_variant",
        "question",
        "answer",
        "expected_local_path",
        "expected_source_url",
        "question_type",
        "anchor_window_index",
        "anchor_start_word",
        "anchor_end_word",
        "anchor_start_char",
        "anchor_end_char",
        "source",
        "attachment_title",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="Generate paired retrieval QA candidates.")
    parser.add_argument("--cache-path", default="benchmarking/raw_attachments")
    parser.add_argument("--output", default="benchmarking/qa_sets/retrieval_qa_paired_candidates.csv")
    parser.add_argument("--source", choices=("both", "docdb", "indico"), default="both")
    parser.add_argument("--max-attachments", type=int, default=20)
    parser.add_argument("--overview-chars", type=int, default=5000)
    parser.add_argument("--target-window-chars", type=int, default=1500)
    parser.add_argument("--windows-per-attachment", type=int, default=1)
    parser.add_argument("--random-seed", type=int, default=20260710)
    parser.add_argument("--model", default=LLM_MODEL)
    parser.add_argument("--api-url", default=normalize_chat_url(LITELLM_API_URL))
    parser.add_argument("--api-key", default=LITELLM_API_KEY)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--fallback-on-error", action="store_true")
    parser.add_argument("--include-html", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Append to an existing CSV and skip completed window IDs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_path = resolve_path(args.cache_path)
    output = resolve_path(args.output)
    extractor = LocalTextExtractor()
    rows_written = initialize_output(output, overwrite=not args.resume)
    done_windows = completed_window_ids(output) if args.resume else set()
    attachments_seen = 0
    window_count = 0

    for record in iter_manifest(cache_path, args.source):
        if should_skip_record(record, args.include_html):
            print(f"[skip] html attachment: {record.get('path')}")
            continue
        if attachments_seen >= args.max_attachments:
            break
        attachments_seen += 1

        text = extract_text(record, extractor)
        if not text:
            print(f"[skip] no text extracted: {record.get('path')}")
            continue

        overview = text[: args.overview_chars]
        seed_basis = str(record.get("path") or record.get("attachment_url") or record.get("filename") or attachments_seen)
        for window in make_windows(
            text,
            args.target_window_chars,
            args.windows_per_attachment,
            seed_basis,
            args.random_seed,
        ):
            window_count += 1
            window_pair_id = f"win_{window_count:04d}"
            if window_pair_id in done_windows:
                print(f"[skip] completed {window_pair_id}")
                continue
            try:
                pair = call_litellm(record, overview, window["text"], args)
            except Exception as exc:
                print(f"[warn] generation failed for {record.get('path')} window={window['window_index']}: {exc}")
                if args.fallback_on_error:
                    pair = lexical_fallback(record)
                else:
                    continue

            title = record.get("title") or record.get("event_title") or record.get("attachment_title") or record.get("filename", "")
            window_rows = []
            for variant in VARIANTS:
                item = pair[variant]
                rows_written += 1
                window_rows.append(
                    {
                        "id": f"qa_{rows_written:04d}",
                        "window_pair_id": window_pair_id,
                        "query_variant": variant,
                        "question": item["question"],
                        "answer": item["answer"],
                        "expected_local_path": record.get("path", ""),
                        "expected_source_url": record.get("attachment_url", ""),
                        "question_type": item["question_type"],
                        "anchor_window_index": window["window_index"],
                        "anchor_start_word": window["anchor_start_word"],
                        "anchor_end_word": window["anchor_end_word"],
                        "anchor_start_char": window["anchor_start_char"],
                        "anchor_end_char": window["anchor_end_char"],
                        "source": record.get("_source_name", ""),
                        "attachment_title": title,
                    }
                )
            append_rows(window_rows, output)
            print(f"[ok] {window_pair_id}: {record.get('path')} window={window['window_index']} rows={rows_written}")

    print(f"Wrote {rows_written} paired QA rows from {window_count} windows to {output}")


if __name__ == "__main__":
    main()
