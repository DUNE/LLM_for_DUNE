"""Shared retrieval scoring utilities for embedding and retriever benchmarks."""

from __future__ import annotations

from typing import Mapping


ERS_WEIGHTS = {
    "path_at_k": 0.24,
    "anchor_at_k": 0.36,
    "mrr_path": 0.12,
    "mrr_anchor": 0.18,
    "best_ov": 0.10,
}

ERS_FORMULA = (
    "0.24*path@k + 0.36*anchor@k + "
    "0.12*mrr_path + 0.18*mrr_anchor + 0.10*best_ov"
)
RERS_FORMULA = "ERS - 0.01*final_top_k"


def compute_ers(metrics: Mapping[str, float]) -> float:
    return sum(ERS_WEIGHTS[key] * float(metrics[key]) for key in ERS_WEIGHTS)


def compute_rers(metrics: Mapping[str, float], final_top_k: int) -> float:
    return compute_ers(metrics) - 0.01 * final_top_k
