#!/usr/bin/env python
"""
Compare Best Model Test-set Results
==================================
This utility reads the consolidated CSV produced by `hyperparam_search.py`
(`experiments/*/final_test_metrics.csv`) and prints a nicely formatted table
of MAE, MSE, ADE and FDE for each architecture.  Optionally it can also render
simple bar plots for a visual comparison.

Usage
-----
    python compare_test_results.py <path_to_final_test_metrics.csv>

If the path is omitted, the tool attempts to locate the most recent
`final_test_metrics.csv` under the `experiments/` directory.
"""
from __future__ import annotations

# fmt: off
import argparse
import glob
import os
from pathlib import Path
from typing import List, Dict

import pandas as pd
import matplotlib.pyplot as plt
# fmt: on


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def find_latest_metrics_file() -> Path | None:
    """Return the newest consolidated metrics CSV, if any."""
    candidates = glob.glob("experiments/**/final_test_metrics.csv", recursive=True)
    if not candidates:
        return None
    candidates.sort(key=lambda p: Path(p).stat().st_mtime, reverse=True)
    return Path(candidates[0])


DEFAULT_SEARCH_DIR = Path("/tscc/nfs/home/bax001/restricted/CSE251B/experiments/search_final")


def find_latest_search_dir() -> Path | None:
    """Locate the most appropriate search_final directory.

    Priority:
        1. Hard-coded cluster path provided by the user (if it exists).
        2. Newest *search_final* folder under any experiments/ subtree relative to CWD.
    """
    # 1) Check explicit absolute directory first
    if DEFAULT_SEARCH_DIR.exists():
        return DEFAULT_SEARCH_DIR

    # 2) Fallback glob search relative to current working dir
    candidates = glob.glob("**/experiments/**/search_final", recursive=True)
    if not candidates:
        return None
    candidates.sort(key=lambda p: Path(p).stat().st_mtime, reverse=True)
    return Path(candidates[0])


def gather_best_from_search_dir(search_dir: Path) -> pd.DataFrame:
    """Parse every model sub-folder under ``search_dir`` and pick the row with the
    lowest validation MSE from its corresponding search_results.csv.

    The function returns one row per model architecture with the following columns
    (assuming they exist in the CSV):
        model_type, hidden_dim, num_layers, val_MSE, val_MSE_unnorm, MAE, MSE, ADE, FDE
    """

    if not search_dir.exists():
        raise FileNotFoundError(f"Search directory '{search_dir}' does not exist.")

    records: List[Dict[str, object]] = []
    for model_dir in sorted(search_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        csv_path = model_dir / "search_results.csv"
        if not csv_path.exists():
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            print(f"[WARN] Could not read {csv_path}: {exc}")
            continue

        if df.empty or "val_MSE" not in df.columns:
            print(f"[WARN] {csv_path} is empty or missing 'val_MSE'. Skipping.")
            continue

        best_idx = df["val_MSE"].idxmin()
        best_row = df.loc[best_idx]
        records.append(best_row.to_dict())

    if not records:
        raise RuntimeError(
            f"No valid search_results.csv files found under {search_dir}. "
            "Have you run the hyper-parameter search script?"
        )

    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare best-model results across architectures.")

    parser.add_argument(
        "metrics_csv",
        nargs="?",
        default=None,
        help="Path to consolidated final_test_metrics.csv. If omitted, the script "
        "builds the summary on the fly from each model's search_results.csv.",
    )

    parser.add_argument(
        "--search_dir",
        default=None,
        help=(
            "Root directory containing model sub-folders with search_results.csv. "
            "If omitted, the script first checks the default cluster path "
            f"({DEFAULT_SEARCH_DIR}), falling back to the newest '*search_final' found."
        ),
    )

    parser.add_argument("--plot", action="store_true", help="Generate bar plots of the metrics.")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Decide input source (consolidated CSV vs. dynamic aggregation)
    # ------------------------------------------------------------------

    if args.metrics_csv:
        csv_path = Path(args.metrics_csv)
        if not csv_path.exists():
            raise FileNotFoundError(f"Provided metrics_csv '{csv_path}' does not exist.")
        df = pd.read_csv(csv_path)
    else:
        # Build DataFrame from individual search_results.csv files
        search_dir = Path(args.search_dir) if args.search_dir else find_latest_search_dir()
        if search_dir is None:
            raise FileNotFoundError(
                "Could not determine search directory. Please pass --search_dir explicitly "
                "or ensure 'experiments/**/search_final' exists."
            )

        df = gather_best_from_search_dir(search_dir)

        # Persist for convenience so that future runs can pick it up quickly
        csv_path = search_dir / "final_test_metrics.csv"
        df.to_csv(csv_path, index=False)

    # ------------------------------------------------------------------
    # Present results
    # ------------------------------------------------------------------

    print("\n=== Best Hyper-parameters & Metrics per Architecture ===")
    print(df.to_string(index=False))

    # ------------------------------------------------------------------
    # Optional plotting
    # ------------------------------------------------------------------

    if args.plot:
        # Create plot output directory next to the source CSV (whichever one we used)
        out_dir = csv_path.parent / "plots"
        out_dir.mkdir(exist_ok=True)

        metrics = ["MAE", "MSE", "ADE", "FDE"]
        for metric in metrics:
            plt.figure(figsize=(8, 5))
            plt.bar(df["model_type"].str.upper(), df[metric])
            plt.title(f"{metric} Comparison (Internal Test)")
            plt.ylabel(metric)
            plt.xlabel("Model Architecture")
            plt.tight_layout()
            plt.savefig(out_dir / f"{metric.lower()}_comparison.png", dpi=300)
            plt.close()

        print(f"Plots saved to {out_dir}\n")


if __name__ == "__main__":
    main() 