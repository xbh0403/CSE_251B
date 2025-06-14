#!/usr/bin/env python
"""
plot_example_trajectories.py
============================
Draws a grid of trajectories (history + prediction + ground-truth) for the best
checkpoint of each architecture.  For every model we randomly choose **five**
examples from each split created during hyper-parameter search (train / val /
internal-test) and arrange them in a 3×5 figure:

Row 0 → Train examples
Row 1 → Validation examples
Row 2 → Test examples

Colours / markers
    • history ........ black solid
    • origin ......... red dot (last history point)
    • prediction ..... blue line with circles
    • ground truth ... green dashed line

Usage
-----
python plot_example_trajectories.py --exp-root /path/to/experiments/search_final \
                                   [--model lstm gru]
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
import sys
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F

from data_utils.data_utils import TrajectoryDataset
from models.model import (
    Seq2SeqLSTMModel,
    Seq2SeqGRUModel,
    Seq2SeqTransformerModel,
    SocialGRUModel,
    MultiModalGRUModel,
)
from models.predict import generate_multimodal_predictions

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------


def create_model(model_type: str, hidden_dim: int, num_layers: int):
    kwargs = dict(input_dim=6, hidden_dim=hidden_dim, output_seq_len=60, output_dim=2, num_layers=num_layers)
    if model_type == "lstm":
        return Seq2SeqLSTMModel(**kwargs)
    elif model_type == "gru":
        return Seq2SeqGRUModel(**kwargs)
    elif model_type == "transformer":
        return Seq2SeqTransformerModel(num_heads=4, **kwargs)
    elif model_type == "social_gru":
        return SocialGRUModel(**kwargs)
    elif model_type == "multimodal_gru":
        return MultiModalGRUModel(num_modes=3, **kwargs)
    else:
        raise ValueError(model_type)


def build_dataloaders(batch_size: int, scale_pos: float, scale_head: float, scale_vel: float):
    """Re-create the 70/15/15 split used in hyperparam_search."""
    if Path("/tscc").exists():
        train_path = "/tscc/nfs/home/bax001/scratch/CSE_251B/data/train.npz"
    else:
        train_path = "data/train.npz"

    full_ds = TrajectoryDataset(train_path, split="train", scale_position=scale_pos, scale_heading=scale_head,
                                scale_velocity=scale_vel, augment=False)
    size = len(full_ds)
    train_size = int(0.7 * size)
    val_size = int(0.15 * size)
    test_size = size - train_size - val_size
    train_ds, val_ds, test_ds = random_split(full_ds, [train_size, val_size, test_size],
                                             generator=torch.Generator().manual_seed(42))
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=True),
        DataLoader(test_ds, batch_size=batch_size, shuffle=True),
    )


def select_best_worst(loader: DataLoader, model, num_each: int = 2) -> Tuple[List[dict], List[dict]]:
    """Return (best,num_each) and (worst,num_each) examples by MAE (unnorm)."""
    best: List[Tuple[float, dict]] = []
    worst: List[Tuple[float, dict]] = []
    with torch.no_grad():
        for batch in loader:
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(DEVICE)

            # predictions
            if hasattr(model, "num_modes"):
                preds_all, conf = model(batch, teacher_forcing_ratio=0.0)
                best_mode = torch.argmax(conf, dim=1)
                idx = best_mode.view(-1, 1, 1, 1).expand(-1, preds_all.size(1), 1, preds_all.size(-1))
                preds = torch.gather(preds_all, dim=2, index=idx).squeeze(2)
            else:
                preds = model(batch, teacher_forcing_ratio=0.0)

            # un-normalise and compute MAE per sample
            scale = batch["scale_position"].view(-1, 1, 1)
            preds_un = preds * scale
            gt_un = batch["future"] * scale
            mae_per_sample = torch.mean(torch.abs(preds_un - gt_un), dim=(1, 2)).cpu().numpy()

            for i in range(preds.size(0)):
                item = {k: v[i] if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                err = float(mae_per_sample[i])
                # insert into best list (keep sorted)
                best.append((err, item))
                best.sort(key=lambda x: x[0])
                if len(best) > num_each:
                    best.pop()
                # worst
                worst.append((err, item))
                worst.sort(key=lambda x: -x[0])
                if len(worst) > num_each:
                    worst.pop()
    return [i[1] for i in best], [i[1] for i in worst]


def denorm_positions(t: torch.Tensor, scale: float, origin: torch.Tensor) -> np.ndarray:
    return (t * scale).cpu().numpy() + origin.cpu().numpy()


def plot_grid(model, split_examples: Tuple[List[dict], List[dict], List[dict]], model_label: str, out_path: Path):
    n_cols = 4
    fig, axes = plt.subplots(3, n_cols, figsize=(n_cols * 4, 12))
    rows = ["Train", "Val", "Test"]

    for row_idx, (row_name, examples) in enumerate(zip(rows, split_examples)):
        for col_idx, ex in enumerate(examples):
            ax = axes[row_idx, col_idx]
            scale = ex["scale_position"].item()
            origin = ex["origin"]
            hist = denorm_positions(ex["history"][0, :, :2], scale, origin)
            ax.plot(hist[:, 0], hist[:, 1], "k-", label="History" if col_idx == 0 else None)
            ax.scatter(hist[-1, 0], hist[-1, 1], c="red", s=20, label="Origin" if col_idx == 0 else None)

            # prediction
            with torch.no_grad():
                batch_single = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in ex.items()}
                if hasattr(model, "num_modes"):
                    preds_all, conf = model(batch_single, teacher_forcing_ratio=0.0)
                    best = torch.argmax(conf, dim=1)
                    preds = preds_all[torch.arange(preds_all.size(0)), :, best, :].squeeze(2)
                else:
                    preds = model(batch_single, teacher_forcing_ratio=0.0)
            preds_np = denorm_positions(preds.squeeze(0), scale, origin)
            ax.plot(preds_np[:, 0], preds_np[:, 1], "bo-", label="Prediction" if col_idx == 0 else None)

            # ground truth (not available for competition test set but our internal split has it)
            if "future" in ex:
                fut = denorm_positions(ex["future"], scale, origin)
                ax.plot(fut[:, 0], fut[:, 1], "g--", label="Ground Truth" if col_idx == 0 else None)

            # --- keep plot square and centred ---
            # gather all coordinates drawn in this axis
            xs = np.concatenate([hist[:, 0], preds_np[:, 0], fut[:, 0] if "future" in ex else np.array([])])
            ys = np.concatenate([hist[:, 1], preds_np[:, 1], fut[:, 1] if "future" in ex else np.array([])])

            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            rng_x = x_max - x_min
            rng_y = y_max - y_min
            side = max(rng_x, rng_y)
            # pad 5% for aesthetics
            pad = side * 0.05
            # center each axis to have equal range
            ax.set_xlim((x_min + x_max) / 2 - side / 2 - pad, (x_min + x_max) / 2 + side / 2 + pad)
            ax.set_ylim((y_min + y_max) / 2 - side / 2 - pad, (y_min + y_max) / 2 + side / 2 + pad)
            ax.set_aspect('equal', 'box')

            # keep ticks for reference, but avoid redundant labels
            if row_idx == 2:
                ax.set_xlabel('X')
            else:
                ax.set_xlabel('')
            if col_idx == 0:
                ax.set_ylabel('Y')
            else:
                ax.set_ylabel('')

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle(f"{model_label} – Trajectories (best and worst)", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot random trajectory examples for best models.")
    parser.add_argument("--exp-root", required=True)
    parser.add_argument("--model", nargs="+", help="Subset of architectures to plot")
    parser.add_argument("--batch", type=int, default=256, help="Loader batch size while sampling")
    args = parser.parse_args()

    exp_root = Path(args.exp_root).resolve()
    summary_file = exp_root / "final_test_metrics.csv"
    if not summary_file.exists():
        sys.exit("final_test_metrics.csv not found.")
    best_df = pd.read_csv(summary_file)
    if args.model:
        best_df = best_df[best_df["model_type"].isin(args.model)]

    # constants from training
    SCALE_POS, SCALE_HEAD, SCALE_VEL = 15.0, 1.0, 5.0

    train_loader, val_loader, test_loader = build_dataloaders(args.batch, SCALE_POS, SCALE_HEAD, SCALE_VEL)

    for _, row in best_df.iterrows():
        model_type = row["model_type"]
        hidden_dim, num_layers = int(row["hidden_dim"]), int(row["num_layers"])
        run_name = f"h{hidden_dim}_l{num_layers}"
        model_name = f"{model_type}_{run_name}"
        ckpt_path = exp_root / model_type / run_name / "best_model.pth"
        if not ckpt_path.exists():
            print(f"[warning] {ckpt_path} missing; skipping {model_type}")
            continue

        model = create_model(model_type, hidden_dim, num_layers).to(DEVICE)
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE)["model_state_dict"])
        model.eval()

        splits = []
        for loader in (train_loader, val_loader, test_loader):
            best, worst = select_best_worst(loader, model, 2)
            splits.append(best + worst)  # best first then worst = 4 samples

        out_fig = exp_root / "plots" / "examples" / f"{model_type}_examples.png"
        plot_grid(model, tuple(splits), model_type.upper(), out_fig)
        print(f"Saved example trajectories for {model_type} → {out_fig}")


if __name__ == "__main__":
    main() 