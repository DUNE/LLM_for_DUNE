#!/usr/bin/env python3
"""
Generate retrieval QA candidates from cached local attachments.

The output is a reviewable CSV. It is not meant to be blindly trusted as final
ground truth; human review should promote good rows into the benchmark QA set.
"""

from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config import LITELLM_API_KEY, LITELLM_API_URL, LLM_MODEL
from src.core.local_attachment_processor import LocalTextExtractor


ALLOWED_TYPES = {"fact_lookup", "summary", "definition", "procedure", "comparison", "number_value"}


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
        with manifest.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("error") or not record.get("path"):
                    continue
                record["_source_name"] = source_name
                yield record


def should_skip_record(record: Dict[str, Any], include_html: bool) -> bool:
    content_type = str(record.get("content_type") or "").lower()
    if not include_html and content_type.startswith("text/html"):
        return True
    return False


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
    raw_text = re.sub(r"\s+", " ", raw_text or "").strip()
    return raw_text


def make_windows(text: str, window_chars: int, windows_per_attachment: int) -> List[Dict[str, Any]]:
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
    if windows_per_attachment <= 1:
        return [make_window(0, 1)]

    max_start = max(0, len(text) - window_chars)
    starts = []
    for i in range(windows_per_attachment):
        starts.append(round(max_start * i / (windows_per_attachment - 1)))

    windows: List[Dict[str, Any]] = []
    seen = set()
    for index, start in enumerate(starts, start=1):
        window = make_window(start, index)
        if not window["text"] or window["text"] in seen:
            continue
        seen.add(window["text"])
        windows.append(window)
    return windows


def build_prompt(
    record: Dict[str, Any],
    overview: str,
    target_window: str,
    questions_per_window: int,
) -> List[Dict[str, str]]:
    title = record.get("title") or record.get("event_title") or record.get("attachment_title") or record.get("filename", "")
    source = record.get("_source_name", "")
    return [
        {
            "role": "system",
            "content": (
                "You create high-quality retrieval benchmark questions for a DUNE RAG system. "
                "You will receive two sections: ATTACHMENT OVERVIEW and TARGET WINDOW. "
                "Use ATTACHMENT OVERVIEW only for orientation. "
                "Generate questions strictly from TARGET WINDOW. "
                "Do not ask questions that require information outside TARGET WINDOW. "
                "Every question must contain a semantic anchor: a specific named concept, component, metric, "
                "method, system, detector, process, parameter, or experiment detail from TARGET WINDOW. "
                "Avoid vague phrases such as 'the numerical sequence', 'the document', 'the study', "
                "'this analysis', or 'the plot' unless paired with a specific named anchor. "
                "The question should be answerable by a reader who has access to TARGET WINDOW, "
                "but not obvious from general DUNE knowledge. "
                "Do not mention file names, URLs, document IDs, local paths, or metadata. "
                "Avoid broad document-level questions unless TARGET WINDOW itself supports them. "
                "Return only valid JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Source: {source}\n"
                f"Title: {title}\n\n"
                f"ATTACHMENT OVERVIEW:\n{overview}\n\n"
                f"TARGET WINDOW:\n{target_window}\n\n"
                f"Generate {questions_per_window} retrieval question(s) from TARGET WINDOW only. "
                "Return a JSON array. Each item must have keys: question, answer, question_type. "
                "The answer must be concise and fully supported by TARGET WINDOW. "
                "The question must include enough specific semantic detail to locate the relevant passage. "
                "Allowed question_type values: fact_lookup, summary, definition, procedure, comparison, number_value."
            ),
        },
    ]


def parse_json_array(text: str) -> List[Dict[str, Any]]:
    text = text.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*\]", text)
        if not match:
            raise
        data = json.loads(match.group(0))

    if not isinstance(data, list):
        raise ValueError("LLM output was not a JSON array")
    return [item for item in data if isinstance(item, dict)]


def generate_questions(
    record: Dict[str, Any],
    overview: str,
    target_window: str,
    args: argparse.Namespace,
) -> List[Dict[str, str]]:
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    payload = {
        "model": args.model,
        "messages": build_prompt(record, overview, target_window, args.questions_per_window),
        "temperature": args.temperature,
        "stream": False,
    }
    response = requests.post(args.api_url, headers=headers, json=payload, timeout=args.timeout)
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]

    questions = []
    for item in parse_json_array(content):
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        question_type = str(item.get("question_type", "fact_lookup")).strip()
        if not question or not answer:
            continue
        if question_type not in ALLOWED_TYPES:
            question_type = "fact_lookup"
        questions.append({"question": question, "answer": answer, "question_type": question_type})
    return questions


def fallback_question(record: Dict[str, Any], target_window: str = "") -> List[Dict[str, str]]:
    title = record.get("title") or record.get("event_title") or record.get("attachment_title") or record.get("filename", "this attachment")
    return [{"question": f"What are the main points discussed in {title}?", "answer": "", "question_type": "summary"}]


def write_rows(rows: List[Dict[str, str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
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
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="Generate retrieval QA candidates from local cached attachments.")
    parser.add_argument("--cache-path", default="benchmarking/raw_attachments")
    parser.add_argument("--output", default="benchmarking/qa_sets/retrieval_qa_candidates.csv")
    parser.add_argument("--source", choices=("both", "docdb", "indico"), default="both")
    parser.add_argument("--max-attachments", type=int, default=20)
    parser.add_argument("--overview-chars", type=int, default=5000)
    parser.add_argument("--target-window-chars", type=int, default=1500)
    parser.add_argument("--windows-per-attachment", type=int, default=2)
    parser.add_argument("--questions-per-window", type=int, default=1)
    parser.add_argument("--model", default=LLM_MODEL)
    parser.add_argument("--api-url", default=normalize_chat_url(LITELLM_API_URL))
    parser.add_argument("--api-key", default=LITELLM_API_KEY)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--fallback-only", action="store_true", help="Do not call the LLM; generate placeholder rows.")
    parser.add_argument("--fallback-on-error", action="store_true", help="Write placeholder rows if an LLM call fails.")
    parser.add_argument("--include-html", action="store_true", help="Include text/html attachments. By default they are skipped.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_path = resolve_path(args.cache_path)
    output = resolve_path(args.output)
    extractor = LocalTextExtractor()
    rows: List[Dict[str, str]] = []
    attachments_seen = 0

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
        windows = make_windows(text, args.target_window_chars, args.windows_per_attachment)
        for window in windows:
            target_window = window["text"]
            try:
                questions = (
                    fallback_question(record, target_window)
                    if args.fallback_only
                    else generate_questions(record, overview, target_window, args)
                )
            except Exception as exc:
                print(f"[warn] LLM generation failed for {record.get('path')} window={window['window_index']}: {exc}")
                if args.fallback_on_error:
                    questions = fallback_question(record, target_window)
                else:
                    continue

            for question in questions[: args.questions_per_window]:
                qa_id = f"qa_{len(rows) + 1:04d}"
                rows.append(
                    {
                        "id": qa_id,
                        "question": question["question"],
                        "answer": question["answer"],
                        "expected_local_path": record.get("path", ""),
                        "expected_source_url": record.get("attachment_url", ""),
                        "question_type": question["question_type"],
                        "anchor_window_index": window["window_index"],
                        "anchor_start_word": window["anchor_start_word"],
                        "anchor_end_word": window["anchor_end_word"],
                        "anchor_start_char": window["anchor_start_char"],
                        "anchor_end_char": window["anchor_end_char"],
                    }
                )

    write_rows(rows, output)
    print(f"Wrote {len(rows)} QA candidate rows to {output}")


if __name__ == "__main__":
    main()
