#!/usr/bin/env python3
"""Plot embedding-layer ERS benchmark results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


MODEL_ORDER = ["BGE", "e5-small-v2", "allMiniLM", "MPNet"]
PALETTE = {
    "BGE": "#0072B2",
    "e5-small-v2": "#009E73",
    "allMiniLM": "#D55E00",
    "MPNet": "#CC79A7",
}


def load_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    numeric = [
        "rank",
        "chunk_size",
        "chunk_overlap",
        "top_k",
        "path_at_k",
        "anchor_at_k",
        "mrr_path",
        "mrr_anchor",
        "best_ov",
        "lat_ms",
        "ERS",
    ]
    for column in numeric:
        df[column] = pd.to_numeric(df[column])
    df["combo_label"] = (
        df["rank"].astype(int).astype(str)
        + ". "
        + df["model"].astype(str)
        + "\n"
        + df["chunk_size"].astype(int).astype(str)
        + "/"
        + df["chunk_overlap"].astype(int).astype(str)
    )
    return df.sort_values("ERS", ascending=False).reset_index(drop=True)


def plot_ers_ranking(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(15, 7))
    colors = [PALETTE.get(model, "#666666") for model in df["model"]]
    ax.bar(range(len(df)), df["ERS"], color=colors, width=0.82)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["combo_label"], rotation=65, ha="right", fontsize=8)
    ax.set_ylabel("ERS")
    ax.set_title("Embedding Combo Ranking by ERS")
    ax.set_ylim(max(0, df["ERS"].min() - 0.03), min(1.0, df["ERS"].max() + 0.03))
    ax.grid(axis="y", alpha=0.25)
    handles = [
        plt.Line2D([0], [0], marker="s", color="w", label=model, markerfacecolor=color, markersize=10)
        for model, color in PALETTE.items()
        if model in set(df["model"])
    ]
    ax.legend(handles=handles, title="Model", frameon=False, ncol=4, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_heatmaps(df: pd.DataFrame, out: Path) -> None:
    models = [model for model in MODEL_ORDER if model in set(df["model"])]
    fig, axes = plt.subplots(1, len(models), figsize=(4.3 * len(models), 4), sharey=True)
    if len(models) == 1:
        axes = [axes]
    vmin, vmax = df["ERS"].min(), df["ERS"].max()
    for ax, model in zip(axes, models):
        pivot = (
            df[df["model"] == model]
            .pivot(index="chunk_size", columns="chunk_overlap", values="ERS")
            .sort_index(ascending=True)
        )
        sns.heatmap(
            pivot,
            ax=ax,
            annot=True,
            fmt=".3f",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            cbar=ax is axes[-1],
            cbar_kws={"label": "ERS"},
        )
        ax.set_title(model)
        ax.set_xlabel("Chunk Overlap")
        ax.set_ylabel("Chunk Size" if ax is axes[0] else "")
    fig.suptitle("ERS by Chunk Size and Overlap", y=1.03)
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_quality_latency(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    for model, group in df.groupby("model"):
        ax.scatter(
            group["lat_ms"],
            group["ERS"],
            s=90,
            alpha=0.88,
            label=model,
            color=PALETTE.get(model, "#666666"),
            edgecolor="white",
            linewidth=0.8,
        )
        for _, row in group.iterrows():
            ax.annotate(
                f"{int(row['chunk_size'])}/{int(row['chunk_overlap'])}",
                (row["lat_ms"], row["ERS"]),
                textcoords="offset points",
                xytext=(5, 4),
                fontsize=8,
            )
    ax.set_xlabel("Mean Latency (ms)")
    ax.set_ylabel("ERS")
    ax.set_title("Retrieval Quality vs Latency")
    ax.grid(alpha=0.25)
    ax.legend(title="Model", frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_best_model_metric_breakdown(df: pd.DataFrame, out: Path) -> None:
    metrics = ["path_at_k", "anchor_at_k", "mrr_path", "mrr_anchor", "best_ov"]
    best = (
        df.sort_values("ERS", ascending=False)
        .groupby("model", sort=False, as_index=False)
        .first()
        .sort_values("ERS", ascending=False)
    )
    best["label"] = best.apply(
        lambda row: (
            f"{row['model']}\n"
            f"chunk {int(row['chunk_size'])}\n"
            f"overlap {int(row['chunk_overlap'])}\n"
            f"ERS {row['ERS']:.3f}"
        ),
        axis=1,
    )
    melted = best.melt(id_vars=["label"], value_vars=metrics, var_name="metric", value_name="score")
    fig, ax = plt.subplots(figsize=(11.5, 7.2))
    sns.barplot(data=melted, x="label", y="score", hue="metric", ax=ax)
    ax.set_ylabel("Score")
    ax.set_xlabel("")
    ax.set_title("Metric Breakdown for Each Model's Best Combo")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", labelsize=9)
    ax.legend(title="Metric", frameon=False, ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.28))
    fig.subplots_adjust(top=0.88, bottom=0.28)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot embedding ERS benchmark results.")
    parser.add_argument("--input", default="benchmarking/results/embedding_layer_24combo_paired_qa_ers.csv")
    parser.add_argument("--output-dir", default="benchmarking/plots")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(input_path)
    plot_ers_ranking(df, output_dir / "embedding_ers_ranking.png")
    plot_heatmaps(df, output_dir / "embedding_ers_heatmap.png")
    plot_quality_latency(df, output_dir / "embedding_quality_latency.png")
    plot_best_model_metric_breakdown(df, output_dir / "embedding_best_model_metric_breakdown.png")

    print(f"Wrote plots to {output_dir}")


if __name__ == "__main__":
    main()
