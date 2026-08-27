#!/usr/bin/env python3
"""Plot retriever-layer Group A no-reranker benchmark results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PALETTE = {
    "dense": "#0072B2",
    "bm25": "#D55E00",
    "hybrid": "#009E73",
}


def load_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    numeric = [
        "rank",
        "final_top_k",
        "path_at_k",
        "anchor_at_k",
        "mrr_path",
        "mrr_anchor",
        "best_ov",
        "lat_ms",
        "ERS",
        "RERS",
    ]
    for column in numeric:
        df[column] = pd.to_numeric(df[column])
    df["hybrid_weight_dense"] = pd.to_numeric(df["hybrid_weight_dense"], errors="coerce")
    df["combo_label"] = df["combo_name"].str.replace("_", "\n", regex=False)
    return df.sort_values("RERS", ascending=False).reset_index(drop=True)


def plot_rers_ranking(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 6.5))
    colors = [PALETTE.get(mode, "#666666") for mode in df["retriever_mode"]]
    ax.bar(range(len(df)), df["RERS"], color=colors, width=0.82)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["combo_label"], rotation=50, ha="right", fontsize=8)
    ax.set_ylabel("RERS")
    ax.set_title("Retriever Group A Ranking by RERS")
    ax.set_ylim(max(0, df["RERS"].min() - 0.03), min(1.0, df["RERS"].max() + 0.03))
    ax.grid(axis="y", alpha=0.25)
    handles = [
        plt.Line2D([0], [0], marker="s", color="w", label=mode, markerfacecolor=color, markersize=10)
        for mode, color in PALETTE.items()
        if mode in set(df["retriever_mode"])
    ]
    ax.legend(handles=handles, title="Retriever", frameon=False, ncol=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_ers_by_topk(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 6))
    sns.lineplot(
        data=df,
        x="final_top_k",
        y="ERS",
        hue="combo_name",
        style="retriever_mode",
        markers=True,
        dashes=False,
        ax=ax,
    )
    ax.set_title("ERS by Returned Context Size")
    ax.set_xlabel("final_top_k")
    ax.set_ylabel("ERS")
    ax.set_xticks(sorted(df["final_top_k"].unique()))
    ax.grid(alpha=0.25)
    ax.legend(title="Combo", fontsize=7, title_fontsize=8, frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_best_by_mode_breakdown(df: pd.DataFrame, out: Path) -> None:
    metrics = ["path_at_k", "anchor_at_k", "mrr_path", "mrr_anchor", "best_ov", "RERS"]
    best = (
        df.sort_values("RERS", ascending=False)
        .groupby("retriever_mode", sort=False, as_index=False)
        .first()
        .sort_values("RERS", ascending=False)
    )
    best["label"] = best.apply(
        lambda row: f"{row['combo_name']}\nRERS {row['RERS']:.3f}",
        axis=1,
    )
    melted = best.melt(id_vars=["label"], value_vars=metrics, var_name="metric", value_name="score")
    fig, ax = plt.subplots(figsize=(10.5, 6.6))
    sns.barplot(data=melted, x="label", y="score", hue="metric", ax=ax)
    ax.set_title("Best Group A Combo per Retriever Mode")
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Metric", frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.28))
    fig.subplots_adjust(bottom=0.28)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_hybrid_heatmap(df: pd.DataFrame, out: Path) -> None:
    hybrid = df[df["retriever_mode"] == "hybrid"].copy()
    if hybrid.empty:
        return
    pivot = hybrid.pivot(index="hybrid_weight_dense", columns="final_top_k", values="RERS").sort_index()
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", cbar_kws={"label": "RERS"}, ax=ax)
    ax.set_title("Hybrid RERS by Dense Weight and final_top_k")
    ax.set_xlabel("final_top_k")
    ax.set_ylabel("dense weight")
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot retriever Group A results.")
    parser.add_argument(
        "--input",
        default="benchmarking/retriever_runs/e5_small_v2_2000_overlap0_group_a_no_reranker/retriever_group_a_rers_ranked.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmarking/plots/retriever_group_a_no_reranker",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_results(Path(args.input))

    plot_rers_ranking(df, output_dir / "retriever_group_a_rers_ranking.png")
    plot_ers_by_topk(df, output_dir / "retriever_group_a_ers_by_topk.png")
    plot_best_by_mode_breakdown(df, output_dir / "retriever_group_a_best_mode_breakdown.png")
    plot_hybrid_heatmap(df, output_dir / "retriever_group_a_hybrid_heatmap.png")
    print(f"Wrote plots to {output_dir}")


if __name__ == "__main__":
    main()
