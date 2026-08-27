#!/usr/bin/env python3
"""
Cache raw DocDB and Indico source files for benchmarking.

This script intentionally stops before text extraction, chunking, embedding,
and Chroma writes. It downloads raw attachment files plus metadata so repeated
benchmark runs can reuse local files.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import parse_qs, urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from requests.auth import HTTPBasicAuth
from urllib3.util.retry import Retry


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "benchmarking" / "raw_events"
DOCDB_BASE_URL = "https://docs.dunescience.org/cgi-bin/private/ShowDocument?docid="


def load_project_env() -> None:
    load_dotenv(ROOT / ".env")


def safe_name(value: Any, fallback: str = "item") -> str:
    text = str(value or fallback).strip()
    text = re.sub(r"[^\w.\-]+", "_", text)
    text = text.strip("._")
    return text or fallback


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=5,
        read=5,
        connect=5,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD", "OPTIONS"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=20)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/126.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json,text/html;q=0.8,*/*;q=0.5",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
        }
    )
    return session


def guess_filename(url: str, headers: Optional[Dict[str, str]] = None, fallback: str = "attachment") -> str:
    headers = headers or {}
    content_disposition = headers.get("content-disposition", "")
    match = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)', content_disposition, flags=re.I)
    if match:
        return safe_name(match.group(1), fallback)

    query_filename = parse_qs(urlparse(url).query).get("filename", [""])[0]
    if query_filename:
        return safe_name(query_filename, fallback)

    path_name = Path(urlparse(url).path).name
    return safe_name(path_name, fallback)


def prefixed_filename(index: int, filename: str) -> str:
    return f"{index:03d}_{safe_name(filename, f'attachment_{index}')}"


def download_file(
    session: requests.Session,
    url: str,
    destination: Path,
    max_file_bytes: int,
) -> Dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    response = session.get(url, headers=headers, stream=True, timeout=120)
    response.raise_for_status()

    total = 0
    failed = False
    with destination.open("wb") as f:
        try:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                total += len(chunk)
                if total > max_file_bytes:
                    raise RuntimeError(f"File exceeded max size {max_file_bytes} bytes: {url}")
                f.write(chunk)
        except Exception:
            failed = True
            raise
        finally:
            if failed:
                f.close()
                destination.unlink(missing_ok=True)

    return {
        "path": str(destination.relative_to(ROOT)),
        "bytes": total,
        "content_type": response.headers.get("content-type", ""),
        "status_code": response.status_code,
    }


def cache_docdb(args: argparse.Namespace) -> Dict[str, int]:
    username = os.getenv("DUNE_DOCDB_USERNAME")
    password = os.getenv("DUNE_DOCDB_PASSWORD")
    if not username or not password:
        raise RuntimeError("DUNE_DOCDB_USERNAME/DUNE_DOCDB_PASSWORD are required for DocDB caching")

    session = build_session()
    session.auth = HTTPBasicAuth(username, password)

    out_root = args.output / "docdb"
    manifest = out_root / "metadata.jsonl"
    if args.reset_manifest and manifest.exists():
        manifest.unlink()

    stats = {
        "attachment_target": args.docdb_limit,
        "docids_scanned": 0,
        "pages_with_attachments": 0,
        "files_available": 0,
        "files_downloaded": 0,
        "errors": 0,
    }
    start = args.docdb_start
    step = -1 if args.docdb_direction == "older" else 1

    for offset in range(args.docdb_max_scan):
        if stats["files_available"] >= args.docdb_limit:
            break

        docid = start + (offset * step)
        if docid < 1:
            break

        stats["docids_scanned"] += 1
        page_url = f"{DOCDB_BASE_URL}{docid}"
        print(f"[DocDB] Checking docid={docid}")

        try:
            page_response = session.get(page_url, timeout=60)
            if page_response.status_code in (401, 403):
                raise RuntimeError(f"Unauthorized for DocDB docid={docid} status={page_response.status_code}")
            if page_response.status_code != 200:
                print(f"[DocDB] Skipping docid={docid}, status={page_response.status_code}")
                continue

            soup = BeautifulSoup(page_response.text, "html.parser")
            title_tag = soup.find("div", id="DocTitle")
            title = title_tag.get_text(" ", strip=True) if title_tag else ""
            links = []
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if "RetrieveFile?docid=" not in href:
                    continue
                full_url = href if href.startswith("http") else urljoin("https://docs.dunescience.org/cgi-bin/", href)
                links.append(full_url)

            if not links:
                print(f"[DocDB] No attachments for docid={docid}")
                continue

            stats["pages_with_attachments"] += 1
            event_dir = out_root / f"doc_{docid}"
            write_json(
                event_dir / "page_metadata.json",
                {
                    "source": "docdb",
                    "docid": docid,
                    "page_url": page_url,
                    "title": title,
                    "retrieved_at": datetime.now(timezone.utc).isoformat(),
                    "attachment_count": len(links),
                },
            )

            for index, file_url in enumerate(links, start=1):
                if stats["files_available"] >= args.docdb_limit:
                    break

                filename = prefixed_filename(index, guess_filename(file_url, fallback=f"docdb_{docid}_{index}"))
                dest = event_dir / "attachments" / filename
                if dest.exists() and not args.force:
                    print(f"[DocDB] Exists, skipping {dest}")
                    file_info = {"path": str(dest.relative_to(ROOT)), "bytes": dest.stat().st_size, "skipped_existing": True}
                else:
                    print(f"[DocDB] Downloading {file_url}")
                    file_info = download_file(session, file_url, dest, args.max_file_bytes)
                    stats["files_downloaded"] += 1
                stats["files_available"] += 1

                record = {
                    "source": "docdb",
                    "docid": docid,
                    "page_url": page_url,
                    "title": title,
                    "attachment_url": file_url,
                    "filename": filename,
                    **file_info,
                }
                append_jsonl(manifest, record)

        except Exception as exc:
            stats["errors"] += 1
            print(f"[DocDB] ERROR docid={docid}: {exc}")
            append_jsonl(
                manifest,
                {"source": "docdb", "docid": docid, "page_url": page_url, "error": str(exc)},
            )

    write_json(out_root / "summary.json", stats)
    return stats


def collect_indico_attachments(obj: Any) -> List[Dict[str, Any]]:
    found: List[Dict[str, Any]] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            url = value.get("download_url")
            files = value.get("files")
            if not url and isinstance(files, list) and files:
                url = files[0].get("download_url") or files[0].get("url")
            if url and (value.get("filename") or value.get("title") or files):
                found.append(value)

            for key in ("attachments", "folders", "material", "materials", "contributions"):
                child = value.get(key)
                if child is not None:
                    visit(child)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(obj)

    normalized = []
    seen = set()
    for item in found:
        url = item.get("download_url")
        title = item.get("title") or item.get("name") or item.get("filename") or ""
        files = item.get("files")
        if not url and isinstance(files, list) and files:
            url = files[0].get("download_url") or files[0].get("url")
            title = files[0].get("filename") or title
        if not url:
            continue
        if url in seen:
            continue
        seen.add(url)
        normalized.append(
            {
                "download_url": url,
                "filename": item.get("filename") or Path(urlparse(url).path).name or title,
                "title": title,
                "content_type": item.get("content_type", ""),
            }
        )
    return normalized


def get_indico_category_export(
    session: requests.Session,
    base_url: str,
    category_id: str,
    event_limit: Optional[int] = None,
) -> Dict[str, Any]:
    url = f"{base_url}/export/categ/{category_id}.json"
    params = {"limit": max(event_limit, 1)} if event_limit is not None else None
    print(f"[Indico] Fetching category export {url}" + (f" params={params}" if params else ""))
    response = session.get(url, params=params, timeout=120)
    response.raise_for_status()
    return response.json()


def get_indico_event_details(session: requests.Session, base_url: str, event_id: str) -> Dict[str, Any]:
    url = f"{base_url}/export/event/{event_id}.json"
    param_candidates = [
        {"attachments": "1", "contributions": "all"},
        {"attachments": "yes", "contributions": "all"},
        {"attachments": "1", "detail": "contributions"},
        {"attachments": "yes", "detail": "contributions"},
        {"attachments": "1", "contributions": "all", "material": "1"},
        {"attachments": "yes", "contributions": "all", "material": "1"},
    ]
    last_payload: Dict[str, Any] = {}
    for params in param_candidates:
        response = session.get(url, params=params, timeout=120)
        response.raise_for_status()
        payload = response.json()
        last_payload = payload
        results = payload.get("results", [])
        details = results[0] if isinstance(results, list) and results else results if isinstance(results, dict) else {}
        if collect_indico_attachments(details):
            return details
    results = last_payload.get("results", [])
    return results[0] if isinstance(results, list) and results else last_payload


def iter_indico_events(session: requests.Session, base_url: str, root_category: str, max_events: int) -> Iterable[Dict[str, Any]]:
    queue = [str(root_category)]
    seen_categories = set()
    yielded_events = set()

    while queue and len(yielded_events) < max_events:
        category_id = queue.pop(0)
        if category_id in seen_categories:
            continue
        seen_categories.add(category_id)

        category_payload = get_indico_category_export(session, base_url, category_id)
        yield {"_category_payload": category_payload, "_category_id": category_id}

        for category in category_payload.get("additionalInfo", {}).get("eventCategories", []):
            child_id = category.get("id") or category.get("categId") or category.get("categoryId")
            if child_id is not None:
                queue.append(str(child_id))

        payload = get_indico_category_export(session, base_url, category_id, max_events)
        for event in payload.get("results", []):
            event_id = str(event.get("id", ""))
            if not event_id or event_id in yielded_events:
                continue
            yielded_events.add(event_id)
            yield {"_event": event, "_category_id": category_id}
            if len(yielded_events) >= max_events:
                break


def cache_indico(args: argparse.Namespace) -> Dict[str, int]:
    base_url = os.getenv("INDICO_BASE_URL", "https://indico.fnal.gov").rstrip("/")
    category_id = str(os.getenv("INDICO_CATEGORY_ID", "443"))
    api_token = os.getenv("INDICO_API_TOKEN")
    if not api_token:
        raise RuntimeError("INDICO_API_TOKEN is required for Indico caching")

    session = build_session()
    session.headers.update(
        {
            "Authorization": f"Bearer {api_token}",
            "Accept": "application/json,text/html;q=0.8,*/*;q=0.5",
        }
    )

    out_root = args.output / "indico"
    manifest = out_root / "metadata.jsonl"
    if args.reset_manifest and manifest.exists():
        manifest.unlink()

    stats = {
        "attachment_target": args.indico_limit,
        "events_seen": 0,
        "events_with_attachments": 0,
        "files_available": 0,
        "files_downloaded": 0,
        "category_exports": 0,
        "errors": 0,
    }

    for item in iter_indico_events(session, base_url, category_id, args.indico_max_events):
        if stats["files_available"] >= args.indico_limit:
            break

        if "_category_payload" in item:
            cid = item["_category_id"]
            write_json(out_root / f"category_{safe_name(cid)}" / "export_categ.json", item["_category_payload"])
            stats["category_exports"] += 1
            continue

        event = item["_event"]
        event_id = str(event.get("id"))
        stats["events_seen"] += 1
        print(f"[Indico] Event {stats['events_seen']}/{args.indico_max_events}: {event_id} {event.get('title', '')}")
        event_dir = out_root / f"event_{safe_name(event_id)}"

        try:
            details = get_indico_event_details(session, base_url, event_id)
            write_json(event_dir / "event_summary.json", event)
            write_json(event_dir / "event_details.json", details)

            attachments = collect_indico_attachments(details)
            if attachments:
                stats["events_with_attachments"] += 1

            for index, attachment in enumerate(attachments, start=1):
                if stats["files_available"] >= args.indico_limit:
                    break

                url = attachment["download_url"]
                if url.startswith("/"):
                    url = urljoin(base_url, url)
                filename = prefixed_filename(index, safe_name(attachment.get("filename"), f"indico_{event_id}_{index}"))
                if "." not in filename:
                    guessed_ext = mimetypes.guess_extension(attachment.get("content_type", "").split(";")[0].strip())
                    if guessed_ext:
                        filename += guessed_ext
                dest = event_dir / "attachments" / filename
                if dest.exists() and not args.force:
                    print(f"[Indico] Exists, skipping {dest}")
                    file_info = {"path": str(dest.relative_to(ROOT)), "bytes": dest.stat().st_size, "skipped_existing": True}
                else:
                    print(f"[Indico] Downloading {url}")
                    file_info = download_file(session, url, dest, args.max_file_bytes)
                    stats["files_downloaded"] += 1
                stats["files_available"] += 1

                record = {
                    "source": "indico",
                    "event_id": event_id,
                    "category_id": item["_category_id"],
                    "event_title": event.get("title", ""),
                    "event_url": f"{base_url}/event/{event_id}/",
                    "attachment_url": url,
                    "filename": filename,
                    "attachment_title": attachment.get("title", ""),
                    **file_info,
                }
                append_jsonl(manifest, record)

        except Exception as exc:
            stats["errors"] += 1
            print(f"[Indico] ERROR event={event_id}: {exc}")
            append_jsonl(
                manifest,
                {"source": "indico", "event_id": event_id, "category_id": item["_category_id"], "error": str(exc)},
            )

    write_json(out_root / "summary.json", stats)
    return stats


def parse_args() -> argparse.Namespace:
    load_project_env()
    parser = argparse.ArgumentParser(description="Cache raw DocDB and Indico attachments for benchmarking.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source", choices=("both", "docdb", "indico"), default="both")
    parser.add_argument("--docdb-limit", type=int, default=int(os.getenv("DOC_LIMIT_DOCDB", "50")), help="Number of DocDB attachments to cache.")
    parser.add_argument("--docdb-start", type=int, default=int(os.getenv("DDB_START_IDX", "0")))
    parser.add_argument(
        "--docdb-direction",
        choices=("older", "newer"),
        default=os.getenv("DOCDB_DIRECTION", "older"),
        help="Scan older or newer DocDB ids from --docdb-start.",
    )
    parser.add_argument("--docdb-max-scan", type=int, default=int(os.getenv("DOCDB_MAX_SCAN", "500")), help="Maximum DocDB docids to inspect while looking for attachments.")
    parser.add_argument("--indico-limit", type=int, default=int(os.getenv("DOC_LIMIT_INDICO", "50")), help="Number of Indico attachments to cache.")
    parser.add_argument("--indico-max-events", type=int, default=int(os.getenv("INDICO_MAX_EVENTS", "500")), help="Maximum Indico events to inspect while looking for attachments.")
    parser.add_argument("--max-file-mb", type=int, default=100)
    parser.add_argument("--force", action="store_true", help="Re-download files that already exist.")
    parser.add_argument("--reset-manifest", action="store_true", help="Remove metadata.jsonl before writing new records.")
    args = parser.parse_args()
    args.output = args.output if args.output.is_absolute() else ROOT / args.output
    args.max_file_bytes = args.max_file_mb * 1024 * 1024
    return args


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {args.output}")

    summary: Dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": args.source,
        "docdb_limit": args.docdb_limit,
        "docdb_start": args.docdb_start,
        "indico_limit": args.indico_limit,
        "output": str(args.output),
    }

    if args.source in ("both", "docdb"):
        summary["docdb"] = cache_docdb(args)

    if args.source in ("both", "indico"):
        summary["indico"] = cache_indico(args)

    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
