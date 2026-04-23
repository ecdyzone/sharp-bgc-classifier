#!/usr/bin/env python3
"""
04_visualize.py

Generates publication-quality figures:
  Fig 1 — t-SNE of ESM-2 embeddings colored by protein role
  Fig 2 — Confusion matrix heatmap
  Fig 3 — Role distribution by BGC type (stacked bar)

Usage:
    python 04_visualize.py --embeddings data/processed/embeddings.npy \
                           --metadata data/processed/metadata.csv \
                           --predictions results/predictions.csv \
                           --output figures/

Outputs:
    figures/tsne_by_role.png
    figures/confusion_matrix.png
    figures/role_by_bgctype.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Consistent color palette across all figures
ROLE_COLORS = {
    "core":       "#1D9E75",
    "regulatory": "#7F77DD",
    "accessory":  "#EF9F27",
    "transport":  "#378ADD",
    "other":      "#888780",
}


def plot_tsne(embeddings: np.ndarray, metadata: pd.DataFrame, out_path: Path):
    """Fig 1 — t-SNE scatter colored by protein role."""
    print("  Computing t-SNE (this takes ~1–3 min)...")

    X = StandardScaler().fit_transform(embeddings)

    tsne = TSNE(
        n_components=2,
        perplexity=40,
        max_iter=1000,
        random_state=42,
        init="pca",
        learning_rate="auto",
    )
    coords = tsne.fit_transform(X)

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor("white")

    roles = metadata["role"].values
    for role, color in ROLE_COLORS.items():
        mask = roles == role
        if mask.sum() == 0:
            continue
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=color, label=role,
            s=10, alpha=0.65, linewidths=0, rasterized=True,
        )

    ax.set_xlabel("t-SNE 1", fontsize=11)
    ax.set_ylabel("t-SNE 2", fontsize=11)
    ax.set_title(
        "ESM-2 embeddings of MIBiG proteins\n"
        "colored by functional role (Actinobacteria)",
        fontsize=12, pad=12,
    )
    ax.legend(
        title="Role", fontsize=9, title_fontsize=9,
        markerscale=2, framealpha=0.8,
        loc="upper right",
    )
    ax.tick_params(labelsize=9)
    sns.despine(ax=ax)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_confusion_matrix(predictions: pd.DataFrame, out_path: Path):
    """Fig 2 — Normalized confusion matrix heatmap."""
    from sklearn.metrics import confusion_matrix

    classes = sorted(predictions["role"].unique())
    cm = confusion_matrix(
        predictions["role"],
        predictions["predicted_role"],
        labels=classes,
        normalize="true",
    )
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(
        cm_df, annot=True, fmt=".2f",
        cmap="YlOrBr", ax=ax,
        linewidths=0.5, linecolor="#dddddd",
        cbar_kws={"shrink": 0.8},
        vmin=0, vmax=1,
    )
    ax.set_xlabel("Predicted role", fontsize=11)
    ax.set_ylabel("True role", fontsize=11)
    ax.set_title("Protein role classifier — normalized confusion matrix", fontsize=11, pad=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_role_by_bgctype(metadata: pd.DataFrame, out_path: Path):
    """Fig 3 — Stacked bar: protein role distribution per BGC type."""
    # Keep top 8 BGC types for readability
    top_types = metadata["bgc_type"].value_counts().head(8).index
    df = metadata[metadata["bgc_type"].isin(top_types)]

    pivot = (
        df.groupby(["bgc_type", "role"])
        .size()
        .unstack(fill_value=0)
    )
    # Normalize to proportions
    pivot_norm = pivot.div(pivot.sum(axis=1), axis=0)

    roles_present = [r for r in ROLE_COLORS if r in pivot_norm.columns]
    colors = [ROLE_COLORS[r] for r in roles_present]

    fig, ax = plt.subplots(figsize=(10, 5))
    pivot_norm[roles_present].plot(
        kind="bar", stacked=True, ax=ax,
        color=colors, width=0.7, edgecolor="white", linewidth=0.5,
    )
    ax.set_xlabel("BGC type", fontsize=11)
    ax.set_ylabel("Proportion of proteins", fontsize=11)
    ax.set_title("Protein functional role distribution by BGC type", fontsize=11, pad=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=9)
    ax.legend(
        title="Role", bbox_to_anchor=(1.01, 1), loc="upper left",
        fontsize=9, title_fontsize=9,
    )
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate visualization figures.")
    parser.add_argument("--embeddings",   default="data/processed/embeddings.npy")
    parser.add_argument("--metadata",     default="data/processed/metadata.csv")
    parser.add_argument("--predictions",  default="results/predictions.csv")
    parser.add_argument("--output",       default="figures/")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    embeddings  = np.load(args.embeddings)
    metadata    = pd.read_csv(args.metadata)
    predictions = pd.read_csv(args.predictions)

    print(f"\n  Loaded {len(embeddings)} embeddings.")

    print("\n=== Figure 1: t-SNE ===")
    plot_tsne(embeddings, metadata, out_dir / "tsne_by_role.png")

    print("\n=== Figure 2: Confusion matrix ===")
    plot_confusion_matrix(predictions, out_dir / "confusion_matrix.png")

    print("\n=== Figure 3: Role by BGC type ===")
    plot_role_by_bgctype(metadata, out_dir / "role_by_bgctype.png")

    print("\nAll figures saved. Ready for the README.")


if __name__ == "__main__":
    main()
