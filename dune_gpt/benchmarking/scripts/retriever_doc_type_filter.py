"""Document-type quota filter for final retriever outputs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class DocumentTypeRatio:
    raw: str
    quotas: Dict[str, int]

    @property
    def enabled(self) -> bool:
        return bool(self.quotas)

    @property
    def final_k(self) -> int:
        return sum(self.quotas.values())


def normalize_document_type(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"slide", "slides", "ppt", "pptx", "presentation"}:
        return "slides"
    if normalized in {"doc", "docs", "document", "documents", "pdf", "text", "html"}:
        return "document"
    return normalized or "unknown"


def parse_document_type_ratio(raw: str | None) -> DocumentTypeRatio:
    text = str(raw or "").strip()
    if not text:
        return DocumentTypeRatio(raw="", quotas={})

    quotas: Dict[str, int] = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(
                "RETRIEVER_DOCUMENT_TYPE_RATIO must look like "
                "'slides:2,document:2'"
            )
        key, value = part.split(":", 1)
        doc_type = normalize_document_type(key)
        try:
            quota = int(value.strip())
        except ValueError as exc:
            raise ValueError(f"Invalid document type quota '{part}'") from exc
        if quota < 0:
            raise ValueError(f"Document type quota must be non-negative: '{part}'")
        if quota:
            quotas[doc_type] = quotas.get(doc_type, 0) + quota

    return DocumentTypeRatio(raw=text, quotas=quotas)


def document_type_ratio_from_env() -> DocumentTypeRatio:
    load_dotenv(ROOT / ".env")
    return parse_document_type_ratio(os.getenv("RETRIEVER_DOCUMENT_TYPE_RATIO"))


def _record_document_type(record: Dict[str, Any]) -> str:
    metadata = record.get("metadata") or {}
    return normalize_document_type(metadata.get("document_type"))


def apply_document_type_ratio(
    ranked: Iterable[Dict[str, Any]],
    ratio: DocumentTypeRatio,
) -> List[Dict[str, Any]]:
    """Select final results by document-type quota while preserving rank order.

    The input should already be truncated to the candidate top-k requested by the
    experiment. If a quota type is short, the remaining slots are filled by the
    highest-ranked unused records from the input.
    """

    ranked_list = [record.copy() for record in ranked]
    if not ratio.enabled:
        for rank, record in enumerate(ranked_list, start=1):
            record["rank"] = rank
        return ranked_list

    selected: List[Dict[str, Any]] = []
    selected_ids: set[str] = set()

    for doc_type, quota in ratio.quotas.items():
        taken = 0
        for record in ranked_list:
            record_id = str(record.get("document_id", ""))
            if record_id in selected_ids:
                continue
            if _record_document_type(record) != doc_type:
                continue
            selected.append(record)
            selected_ids.add(record_id)
            taken += 1
            if taken >= quota:
                break

    target_count = ratio.final_k
    if len(selected) < target_count:
        for record in ranked_list:
            record_id = str(record.get("document_id", ""))
            if record_id in selected_ids:
                continue
            selected.append(record)
            selected_ids.add(record_id)
            if len(selected) >= target_count:
                break

    selected.sort(key=lambda record: int(record.get("rank") or 10**9))
    for rank, record in enumerate(selected, start=1):
        record["rank"] = rank
    return selected


def ratio_manifest(ratio: DocumentTypeRatio) -> Dict[str, Any]:
    return {
        "enabled": ratio.enabled,
        "raw": ratio.raw,
        "quotas": ratio.quotas,
        "filtered_result_k": ratio.final_k if ratio.enabled else None,
    }
