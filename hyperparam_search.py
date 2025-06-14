#!/usr/bin/env python
"""
Hyperparameter Search Script
===========================
This script performs an exhaustive grid-search over the following hyper-parameters:
    hidden_dim  ∈  {64, 128, 256}
    num_layers  ∈  {1, 2, 3}

For every **model architecture** implemented in `models.model` it:
1. Trains using the enhanced `train_model` / `train_multimodal_model` routines.
2. Logs the validation metrics, keeping track of the configuration that attains
   the lowest *validation MSE*.
3. Evaluates the best checkpoint on an **internal test split** and the official
   **competition test set**.
4. Writes the predictions for the competition test set to a submission CSV
   file located inside that run's directory.
5. Consolidates per-model performance into
   `experiments/final_test_metrics.csv` so that a follow-up comparison script
   can visualise the overall results.

The script is intentionally written to be *self-contained* and to make no
assumptions about being executed from a specific working directory: it always
computes relative paths from its own location.
"""
from __future__ import annotations

import os
import sys
from itertools import product
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, List

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, random_split
import argparse

# Local imports – NB: the project already lives on PYTHONPATH when executed via
# the provided launch scripts; if not, append the repository root.
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from data_utils.data_utils import TrajectoryDataset  # type: ignore
from models.model import (
    Seq2SeqLSTMModel,
    Seq2SeqGRUModel,
    Seq2SeqTransformerModel,
    SocialGRUModel,
    MultiModalGRUModel,
)
from models.train import train_model, train_multimodal_model
from models.metrics import compute_metrics
from models.predict import (
    generate_predictions,
    generate_multimodal_predictions,
    create_submission,
)

# ---------------------------------------------------------------------------
# Hyper-parameter grid & global constants
# ---------------------------------------------------------------------------
HIDDEN_DIMS = [64, 128, 256]
NUM_LAYERS = [1, 2, 3]
BATCH_SIZE = 64
N_EPOCHS = 200
EARLY_STOPPING = 30

SCALE_POSITION = 15.0
SCALE_HEADING = 1.0
SCALE_VELOCITY = 5.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def get_dataset_paths() -> Tuple[str, str]:
    """Return absolute paths to the train & test .npz files depending on host."""
    if os.getcwd().startswith("/tscc"):
        train_path = "/tscc/nfs/home/bax001/scratch/CSE_251B/data/train.npz"
        test_path = "/tscc/nfs/home/bax001/scratch/CSE_251B/data/test_input.npz"
    else:
        train_path = "data/train.npz"
        test_path = "data/test_input.npz"
    return train_path, test_path


def build_dataloaders(batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """Create train / val / internal-test / competition-test loaders."""
    train_path, test_path = get_dataset_paths()

    full_dataset = TrajectoryDataset(
        train_path,
        split="train",
        scale_position=SCALE_POSITION,
        scale_heading=SCALE_HEADING,
        scale_velocity=SCALE_VELOCITY,
        augment=True,
    )
    competition_dataset = TrajectoryDataset(
        test_path,
        split="test",
        scale_position=SCALE_POSITION,
        scale_heading=SCALE_HEADING,
        scale_velocity=SCALE_VELOCITY,
    )

    # 70 / 15 / 15 split for train / val / internal-test
    dataset_size = len(full_dataset)
    train_size = int(0.7 * dataset_size)
    val_size = int(0.15 * dataset_size)
    test_size = dataset_size - train_size - val_size

    train_ds, val_ds, internal_test_ds = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    internal_test_loader = DataLoader(internal_test_ds, batch_size=batch_size, shuffle=False)
    competition_test_loader = DataLoader(competition_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, internal_test_loader, competition_test_loader


def create_model(model_type: str, hidden_dim: int, num_layers: int):
    """Factory that instantiates the requested architecture."""
    kwargs = dict(
        input_dim=6,
        hidden_dim=hidden_dim,
        output_seq_len=60,
        output_dim=2,
        num_layers=num_layers,
    )

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
        raise ValueError(f"Unsupported model_type: {model_type}")


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def evaluate_on_internal_test(model, loader, is_multimodal: bool) -> Dict[str, float]:
    """Compute MAE, MSE, ADE, FDE on the *denormalised* internal test set."""
    model.eval()
    maes, mses, ades, fdes = [], [], [], []

    with torch.no_grad():
        for batch in loader:
            # Move tensors to DEVICE
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(DEVICE)

            # Forward pass (no teacher forcing, multi-modal aware)
            if is_multimodal:
                preds, conf = model(batch, teacher_forcing_ratio=0.0)
                best_idx = torch.argmax(conf, dim=1)
                preds = torch.gather(
                    preds,
                    dim=2,
                    index=best_idx.view(-1, 1, 1, 1).expand(-1, preds.size(1), 1, preds.size(-1)),
                ).squeeze(2)
            else:
                preds = model(batch, teacher_forcing_ratio=0.0)

            # Denormalise positions
            scale_pos = batch["scale_position"].view(-1, 1, 1)
            preds_unnorm = preds * scale_pos
            gt_unnorm = batch["future"] * scale_pos

            metrics = compute_metrics(preds_unnorm.cpu(), gt_unnorm.cpu())
            maes.append(metrics["MAE"])
            mses.append(metrics["MSE"])
            ades.append(metrics["ADE"])
            fdes.append(metrics["FDE"])

    return {
        "MAE": float(np.mean(maes)),
        "MSE": float(np.mean(mses)),
        "ADE": float(np.mean(ades)),
        "FDE": float(np.mean(fdes)),
    }


def run_single_experiment(
    model_type: str,
    hidden_dim: int,
    num_layers: int,
    loaders: Tuple[DataLoader, DataLoader, DataLoader, DataLoader],
    root_dir: Path,
) -> Tuple[float, float, Dict[str, float]]:
    """Train & evaluate a single configuration. Returns (val_loss, val_mse_unnorm, test_metrics)."""
    train_loader, val_loader, internal_test_loader, competition_test_loader = loaders

    run_name = f"h{hidden_dim}_l{num_layers}"
    run_dir = root_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------------
    model = create_model(model_type, hidden_dim, num_layers).to(DEVICE)

    if model_type == "multimodal_gru":
        model, checkpoint, _metrics_hist = train_multimodal_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=N_EPOCHS,
            early_stopping_patience=EARLY_STOPPING,
            lr=1e-3,
            weight_decay=1e-4,
            lr_step_size=20,
            lr_gamma=0.25,
            teacher_forcing_ratio=0.5,
            physics_weight=0.2,
            model_name=f"{model_type}_{run_name}",
            save_logs=True,
        )
    else:
        model, checkpoint, _metrics_hist = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=N_EPOCHS,
            early_stopping_patience=EARLY_STOPPING,
            lr=1e-3,
            weight_decay=1e-4,
            lr_step_size=20,
            lr_gamma=0.25,
            teacher_forcing_ratio=0.5,
            model_name=f"{model_type}_{run_name}",
            save_logs=True,
        )

    # Persist best checkpoint
    ckpt_path = run_dir / "best_model.pth"
    torch.save(checkpoint, ckpt_path)

    # Ensure model weights correspond to the best checkpoint for evaluation
    model.load_state_dict(checkpoint["model_state_dict"])

    # ---------------------------------------------------------------------
    # Validation performance (normalized and unnormalized)
    # ---------------------------------------------------------------------
    val_loss = float(checkpoint["val_loss"])
    val_mse_unnorm = float(checkpoint.get("val_mse_unnorm", np.nan))

    # ---------------------------------------------------------------------
    # Internal test metrics
    # ---------------------------------------------------------------------
    test_metrics = evaluate_on_internal_test(
        model,
        internal_test_loader,
        is_multimodal=(model_type == "multimodal_gru"),
    )

    # ---------------------------------------------------------------------
    # Competition test ‑ submission file
    # ---------------------------------------------------------------------
    if model_type == "multimodal_gru":
        preds = generate_multimodal_predictions(model, competition_test_loader)
    else:
        preds = generate_predictions(model, competition_test_loader)

    submission_path = run_dir / "submission.csv"
    create_submission(preds, output_file=str(submission_path))

    # Free GPU RAM before the next run
    del model
    torch.cuda.empty_cache()

    return val_loss, val_mse_unnorm, test_metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    # Allow users to pick specific architectures via --model flag so they can launch
    # independent jobs in parallel (e.g. one per SBATCH submission).

    parser = argparse.ArgumentParser(description="Grid-search over hidden_dim/num_layers for selected model(s).")
    parser.add_argument(
        "--model", "-m",
        nargs="+",
        help="Model architecture(s) to run. Choices: lstm gru transformer social_gru multimodal_gru. If omitted runs all.",
        choices=["lstm", "gru", "transformer", "social_gru", "multimodal_gru"],
    )
    args = parser.parse_args()

    exp_root = Path("/tscc/nfs/home/bax001/restricted/CSE251B/experiments") / "search_final"
    exp_root.mkdir(parents=True, exist_ok=True)

    loaders = build_dataloaders(BATCH_SIZE)

    # Architectures to evaluate
    if args.model:
        model_types = args.model
    else:
        model_types = [
            "lstm",
            "gru",
            "social_gru",
            "multimodal_gru",
        ]

    # Where we accumulate the best-per-model results
    summary_records: List[Dict[str, object]] = []

    for model_type in model_types:
        print(f"\n=== Hyper-parameter search for {model_type.upper()} ===")
        model_dir = exp_root / model_type
        model_dir.mkdir(exist_ok=True)

        search_results: List[Dict[str, object]] = []

        for hidden_dim, num_layers in product(HIDDEN_DIMS, NUM_LAYERS):
            print(f"\n→ Training {model_type} with h={hidden_dim}, layers={num_layers}")
            val_loss, val_mse_unnorm, test_metrics = run_single_experiment(
                model_type,
                hidden_dim,
                num_layers,
                loaders,
                model_dir,
            )

            record = {
                "model_type": model_type,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "val_MSE": val_loss,
                "val_MSE_unnorm": val_mse_unnorm,
                **test_metrics,
            }
            search_results.append(record)

        # Save per-model search results
        df = pd.DataFrame(search_results)
        df.to_csv(model_dir / "search_results.csv", index=False)

        # Identify best hyper-parameters (lowest validation MSE)
        best_idx = df["val_MSE"].idxmin()
        best_row = df.loc[best_idx]
        summary_records.append(best_row.to_dict())

        print(f"Best {model_type}: h={best_row['hidden_dim']}, layers={best_row['num_layers']} (Val MSE={best_row['val_MSE']:.4f})")

    # ---------------------------------------------------------------------
    # Consolidate / update global summary file
    # ---------------------------------------------------------------------
    new_df = pd.DataFrame(summary_records)
    summary_csv = exp_root / "final_test_metrics.csv"

    if summary_csv.exists():
        prev_df = pd.read_csv(summary_csv)
        # Ensure model_type is unique index for replacement
        prev_df = prev_df.set_index("model_type")
        new_df = new_df.set_index("model_type")
        prev_df.update(new_df)
        summary_df = prev_df.reset_index()
    else:
        summary_df = new_df

    summary_df.to_csv(summary_csv, index=False)

    print("\n=== Final Test-set Performance Summary (up-to-date) ===")
    print(summary_df.to_string(index=False))
    print(f"\nSaved consolidated metrics to {summary_csv}\n")


if __name__ == "__main__":
    main() 