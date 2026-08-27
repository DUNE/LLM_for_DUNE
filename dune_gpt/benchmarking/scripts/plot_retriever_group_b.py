#!/usr/bin/env python3
"""Plot retriever-layer Group B hybrid reranker benchmark results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


RERANKER_LABELS = {
    "cross-encoder/ms-marco-MiniLM-L6-v2": "MiniLM",
    "BAAI/bge-reranker-base": "BGE-base",
    "BAAI/bge-reranker-v2-m3": "BGE-v2-m3",
}


def load_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    numeric = [
        "rank",
        "candidate_k",
        "final_top_k",
        "hybrid_weight_dense",
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
    df["reranker_label"] = df["reranker_model"].map(RERANKER_LABELS).fillna(df["reranker_model"])
    df["combo_label"] = (
        df["reranker_label"]
        + "\n"
        + "w="
        + df["hybrid_weight_dense"].map(lambda value: f"{value:.2f}")
        + "\n"
        + "top="
        + df["final_top_k"].astype(int).astype(str)
    )
    return df.sort_values("RERS", ascending=False).reset_index(drop=True)


def plot_rers_ranking(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 8.5))
    palette = dict(zip(sorted(df["reranker_label"].unique()), sns.color_palette("colorblind", df["reranker_label"].nunique())))
    colors = [palette[label] for label in df["reranker_label"]]
    ax.bar(range(len(df)), df["RERS"], color=colors, width=0.82)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["combo_label"], rotation=0, ha="center", fontsize=7)
    ax.set_ylabel("RERS")
    ax.set_title("Retriever Group B Ranking by RERS")
    ax.set_ylim(max(0, df["RERS"].min() - 0.03), min(1.0, df["RERS"].max() + 0.03))
    ax.grid(axis="y", alpha=0.25)
    handles = [
        plt.Line2D([0], [0], marker="s", color="w", label=label, markerfacecolor=color, markersize=10)
        for label, color in palette.items()
    ]
    ax.legend(handles=handles, title="Reranker", frameon=False, ncol=3, loc="upper right")
    fig.subplots_adjust(left=0.06, right=0.99, top=0.90, bottom=0.26)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_heatmaps(df: pd.DataFrame, out: Path) -> None:
    labels = list(df["reranker_label"].drop_duplicates())
    fig, axes = plt.subplots(1, len(labels), figsize=(4.8 * len(labels), 4.2), sharey=True)
    if len(labels) == 1:
        axes = [axes]
    vmin, vmax = df["RERS"].min(), df["RERS"].max()
    for ax, label in zip(axes, labels):
        pivot = (
            df[df["reranker_label"] == label]
            .pivot(index="hybrid_weight_dense", columns="final_top_k", values="RERS")
            .sort_index()
        )
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".3f",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            cbar=ax is axes[-1],
            cbar_kws={"label": "RERS"},
            ax=ax,
        )
        ax.set_title(label)
        ax.set_xlabel("final_top_k")
        ax.set_ylabel("dense weight" if ax is axes[0] else "")
    fig.suptitle("Group B RERS by Reranker, Dense Weight, and final_top_k", y=1.03)
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_best_reranker_breakdown(df: pd.DataFrame, out: Path) -> None:
    metrics = ["path_at_k", "anchor_at_k", "mrr_path", "mrr_anchor", "best_ov", "RERS"]
    best = (
        df.sort_values("RERS", ascending=False)
        .groupby("reranker_label", sort=False, as_index=False)
        .first()
        .sort_values("RERS", ascending=False)
    )
    best["label"] = best.apply(
        lambda row: (
            f"{row['reranker_label']}\n"
            f"w{row['hybrid_weight_dense']:.2f} top{int(row['final_top_k'])}\n"
            f"RERS {row['RERS']:.3f}"
        ),
        axis=1,
    )
    melted = best.melt(id_vars=["label"], value_vars=metrics, var_name="metric", value_name="score")
    fig, ax = plt.subplots(figsize=(11, 6.8))
    sns.barplot(data=melted, x="label", y="score", hue="metric", ax=ax)
    ax.set_title("Best Group B Combo per Reranker")
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Metric", frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.28))
    fig.subplots_adjust(bottom=0.28)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_ers_rers_tradeoff(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6))
    sns.scatterplot(
        data=df,
        x="ERS",
        y="RERS",
        hue="reranker_label",
        style="final_top_k",
        size="hybrid_weight_dense",
        sizes=(70, 150),
        ax=ax,
    )
    ax.set_title("Group B ERS vs RERS")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot retriever Group B results.")
    parser.add_argument(
        "--input",
        default="benchmarking/retriever_runs/e5_small_v2_2000_overlap0_group_b_hybrid_reranker/retriever_group_b_rers_ranked.csv",
    )
    parser.add_argument("--output-dir", default="benchmarking/plots/retriever_group_b_hybrid_reranker")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_results(Path(args.input))

    plot_rers_ranking(df, output_dir / "retriever_group_b_rers_ranking.png")
    plot_heatmaps(df, output_dir / "retriever_group_b_rers_heatmap.png")
    plot_best_reranker_breakdown(df, output_dir / "retriever_group_b_best_reranker_breakdown.png")
    plot_ers_rers_tradeoff(df, output_dir / "retriever_group_b_ers_rers_tradeoff.png")
    print(f"Wrote plots to {output_dir}")


if __name__ == "__main__":
    main()
