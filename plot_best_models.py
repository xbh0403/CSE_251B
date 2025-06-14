#!/usr/bin/env python
"""
plot_best_models.py
===================
Generate training-curve figures for the best hyper-parameter configuration of
each architecture, together with the final (internal) test-set errors.

The script expects that you have already run
`hyperparam_search.py` so the directory structure is:
    experiments/search_final/
        ├── <model>/h*_l*/               <- run directories
        │       └── <model>_h*_l*_best.pth
        ├── <model>/search_results.csv
        └── final_test_metrics.csv

It loads `metrics_history` from each best checkpoint, plots Train vs Validation
loss (MSE) and Train/Val MAE, and overlays the internal-test MAE/MSE stored in
`search_results.csv`.

Usage
-----
python plot_best_models.py --exp-root /path/to/experiments/search_final

Figures are saved under <exp-root>/plots/best_models/ .
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_best_rows(exp_root: Path) -> pd.DataFrame:
    summary_file = exp_root / "final_test_metrics.csv"
    if not summary_file.exists():
        sys.exit("final_test_metrics.csv not found – run hyperparam_search.py first.")
    return pd.read_csv(summary_file)


def build_checkpoint_path(exp_root: Path, row: pd.Series) -> Path:
    model_type = row["model_type"]
    run_name = f"h{row['hidden_dim']}_l{row['num_layers']}"
    model_name = f"{model_type}_{run_name}"
    ckpt = exp_root / model_type / run_name / f"best_model.pth"
    if not ckpt.exists():
        sys.exit(f"Checkpoint {ckpt} not found – did the run complete?")
    return ckpt


def plot_curves(metrics: Dict[str, list], test_mae: float, test_mse: float, model_label: str, out_dir: Path):
    epochs = metrics["epoch"]

    # --- MSE plot ---
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, metrics["val_mse_unnorm"], label="Val MSE (unnorm)")
    if "train_mse_unnorm" in metrics:
        plt.plot(epochs, metrics["train_mse_unnorm"], label="Train MSE (unnorm)")
    # horizontal line for test MSE
    plt.axhline(test_mse, color="red", linestyle="--", label=f"Test MSE = {test_mse:.3g}")
    plt.xlabel("Epoch")
    plt.ylabel("MSE (m²)")
    plt.title(f"{model_label} – MSE")
    plt.legend()
    plt.grid(True)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_dir / f"{model_label}_mse.png", dpi=300)
    plt.close()

    # --- MAE plot ---
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, metrics["val_mae_unnorm"], label="Val MAE (unnorm)")
    if "train_mae_unnorm" in metrics:
        plt.plot(epochs, metrics["train_mae_unnorm"], label="Train MAE (unnorm)")
    plt.axhline(test_mae, color="red", linestyle="--", label=f"Test MAE = {test_mae:.3g}")
    plt.xlabel("Epoch")
    plt.ylabel("MAE (m)")
    plt.title(f"{model_label} – MAE")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / f"{model_label}_mae.png", dpi=300)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training curves for best hyper-parameter models.")
    parser.add_argument("--exp-root", required=True, help="Path to experiments/search_final directory")
    args = parser.parse_args()

    exp_root = Path(args.exp_root).expanduser().resolve()
    if not exp_root.exists():
        sys.exit(f"{exp_root} does not exist.")

    best_df = load_best_rows(exp_root)
    plot_dir = exp_root / "plots" / "best_models"

    for _, row in best_df.iterrows():
        ckpt_path = build_checkpoint_path(exp_root, row)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        metrics = ckpt.get("metrics_history")
        if metrics is None:
            print(f"[warning] No metrics_history in {ckpt_path}, skipping.")
            continue

        model_label = row["model_type"].upper()
        test_mae = row.get("MAE", float("nan"))
        test_mse = row.get("MSE", float("nan"))

        plot_curves(metrics, test_mae, test_mse, model_label, plot_dir)
        print(f"Plotted curves for {model_label} → {plot_dir}")


if __name__ == "__main__":
    main() 