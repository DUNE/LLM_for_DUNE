#!/usr/bin/env python3
"""Recompute embedding-layer ERS from existing metric CSV files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from retrieval_scores import ERS_FORMULA, compute_ers


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute ERS and ranking for an embedding-layer CSV.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    rows = list(csv.DictReader(input_path.open("r", newline="", encoding="utf-8-sig")))
    if not rows:
        raise RuntimeError(f"No rows found in {input_path}")

    for row in rows:
        row["ERS"] = f"{compute_ers(row):.12f}"

    rows.sort(key=lambda item: float(item["ERS"]), reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = str(rank)

    fieldnames = list(rows[0].keys())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote recomputed ERS ranking: {output_path}")
    print(f"ERS = {ERS_FORMULA.replace('@k', '@5')}")


if __name__ == "__main__":
    main()
