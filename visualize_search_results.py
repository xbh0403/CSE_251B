#!/usr/bin/env python
"""
visualize_search_results.py
===========================
Utility to inspect the outcome of `hyperparam_search.py`.

It offers two complementary views:
1.  Per-architecture hyper-parameter grid as a heat-map (val_MSE_unnorm or val_MSE).
2.  A consolidated table of the best configuration for every model (reads
    the final_test_metrics.csv that the search script keeps up-to-date).

Usage
-----
$ python visualize_search_results.py \
        --exp-root /path/to/experiments/search_final [--metric val_MSE_unnorm]

Requirements: pandas, matplotlib, seaborn (optional; falls back to plain
matplotlib if seaborn isn't available).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle

try:
    import seaborn as sns
    _HAS_SNS = True
except ImportError:  # pragma: no cover
    _HAS_SNS = False

DEFAULT_METRIC = "val_MSE_unnorm"


def load_search_csvs(exp_root: Path) -> List[pd.DataFrame]:
    """Return a list of search_result dataframes (one per architecture)."""
    dfs = []
    for csv_path in exp_root.glob("*/search_results.csv"):
        df = pd.read_csv(csv_path)
        model_type = csv_path.parent.name
        df["model_type"] = model_type
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError(f"No search_results.csv files found under {exp_root}")
    return dfs


def plot_heatmap(df: pd.DataFrame, metric: str, model_type: str, out_dir: Path, std_factor: float):
    pivot = df.pivot(index="num_layers", columns="hidden_dim", values=metric)
    outlier_mask_flat = detect_outliers_z(df, metric, std_factor)

    # Build mask shaped like pivot (True where value is an outlier)
    mask = pd.DataFrame(False, index=pivot.index, columns=pivot.columns)
    for idx, is_out in outlier_mask_flat.items():
        if is_out:
            row = df.loc[idx]
            mask.loc[row['num_layers'], row['hidden_dim']] = True

    plt.figure(figsize=(6, 4))
    cmap_choice = "RdYlGn_r"  # green = low (good), red = high (bad)
    if _HAS_SNS:
        sns.heatmap(pivot, mask=mask, annot=False, cmap=cmap_choice, cbar_kws={"label": metric})
    else:
        shown = pivot.mask(mask)
        plt.imshow(shown.values, aspect="auto", cmap=cmap_choice)
        plt.colorbar(label=metric)

    ax = plt.gca()
    # Draw overlay for outliers and annotate all values
    for y, num_layers in enumerate(pivot.index):
        for x, hidden_dim in enumerate(pivot.columns):
            val = pivot.loc[num_layers, hidden_dim]
            is_out = mask.loc[num_layers, hidden_dim]
            if is_out:
                # grey rectangle
                ax.add_patch(Rectangle((x, y), 1, 1, facecolor='lightgrey', edgecolor='black'))
            txt_color = "red" if is_out else "black"
            ax.text(x + 0.5, y + 0.5, f"{val:.2f}", ha="center", va="center", color=txt_color)
    ax.set_xticks([i + 0.5 for i in range(len(pivot.columns))])
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks([i + 0.5 for i in range(len(pivot.index))])
    ax.set_yticklabels(pivot.index)
    plt.title(f"{model_type.upper()} – {metric}")
    plt.xlabel("hidden_dim")
    plt.ylabel("num_layers")

    # Highlight the cell with the overall minimum value (best result)
    min_val = pivot.min().min()
    min_pos = [(r, c) for r in range(len(pivot.index)) for c in range(len(pivot.columns)) if pivot.iat[r, c] == min_val][0]
    rect = Rectangle(min_pos, 1, 1, fill=False, edgecolor="blue", linewidth=2)
    ax.add_patch(rect)

    out_dir.mkdir(exist_ok=True, parents=True)
    plt.tight_layout()
    plt.savefig(out_dir / f"{model_type}_{metric}_heatmap.png", dpi=300)
    plt.close()


# ---------------------------------------------------------------------------
# Outlier handling
# ---------------------------------------------------------------------------


def detect_outliers_z(df: pd.DataFrame, column: str, std_factor: float = 2.5):
    """Return boolean Series marking rows whose column value > mean + k·std."""
    if column not in df.columns:
        return pd.Series(False, index=df.index)
    mu = df[column].mean()
    sigma = df[column].std()
    threshold = mu + std_factor * sigma
    return df[column] > threshold


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise hyper-parameter search results.")
    parser.add_argument("--exp-root", type=str, required=True, help="Path to experiments/search_* directory")
    parser.add_argument("--metric", default=DEFAULT_METRIC, help="Metric to visualise in heat-maps (column name in CSV)")
    parser.add_argument("--no-heatmap", action="store_true", help="Skip heat-map rendering, only print tables")
    args = parser.parse_args()

    exp_root = Path(args.exp_root).expanduser().resolve()
    if not exp_root.exists():
        sys.exit(f"Experiment root {exp_root} does not exist.")

    dfs = load_search_csvs(exp_root)

    print("\n================ Hyper-parameter Grid Results ================")
    for df in dfs:
        model_type = df["model_type"].iloc[0]
        print(f"\n--- {model_type.upper()} ---")
        print(df.sort_values(args.metric if args.metric in df.columns else "val_MSE")[
            ["hidden_dim", "num_layers", args.metric if args.metric in df.columns else "val_MSE"]
        ].to_string(index=False))

        if not args.no_heatmap and args.metric in df.columns:
            plot_heatmap(df, args.metric, model_type, exp_root / "plots", std_factor=2.5)

    # ------------------------------------------------------------------
    # Show consolidated best results (if available)
    # ------------------------------------------------------------------
    summary_file = exp_root / "final_test_metrics.csv"
    if summary_file.exists():
        best_df = pd.read_csv(summary_file)
        print("\n================ Best-of-each-model Summary ================")
        print(best_df.to_string(index=False))
    else:
        print("\n[warning] final_test_metrics.csv not found – have you completed the sweep?")

    if not args.no_heatmap and (exp_root / "plots").exists():
        print(f"\nHeat-maps saved under {exp_root / 'plots'}\n")


if __name__ == "__main__":
    main() 