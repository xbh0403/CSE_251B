# Modified main.py to use the enhanced training functions with metric logging

import torch
import numpy as np
import os
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import matplotlib.pyplot as plt

from data_utils.data_utils import TrajectoryDataset
from models.model import Seq2SeqLSTMModel, Seq2SeqGRUModel, Seq2SeqTransformerModel, SocialGRUModel, MultiModalGRUModel
from models.train import train_model, train_multimodal_model, save_training_metrics
from models.predict import generate_predictions, create_submission, generate_multimodal_predictions
from models.metrics import compute_metrics, visualize_predictions

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Data paths (update these to your actual paths)
    if os.getcwd().startswith('/tscc'):
        train_path = '/tscc/nfs/home/bax001/scratch/CSE_251B/data/train.npz'
        test_path = '/tscc/nfs/home/bax001/scratch/CSE_251B/data/test_input.npz'
    else:
        train_path = 'data/train.npz'
        test_path = 'data/test_input.npz'
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Hyperparameters
    scale_position = 15.0
    scale_heading = 1.0
    scale_velocity = 5.0
    batch_size = 64
    hidden_dim = 128
    num_layers = 2
    teacher_forcing_ratio = 0.5
    
    # Create the full dataset
    print("Creating datasets...")
    full_train_dataset = TrajectoryDataset(train_path, split='train', scale_position=scale_position, scale_heading=scale_heading, scale_velocity=scale_velocity, augment=True)
    competition_test_dataset = TrajectoryDataset(test_path, split='test', scale_position=scale_position, scale_heading=scale_heading, scale_velocity=scale_velocity)
    
    # Create train/validation/test split with 70/15/15 ratio
    dataset_size = len(full_train_dataset)
    train_size = int(0.7 * dataset_size)
    val_size = int(0.15 * dataset_size)
    test_size = dataset_size - train_size - val_size  # remaining 15%
    
    # Create the splits using random_split
    train_dataset, val_dataset, internal_test_dataset = random_split(
        full_train_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    internal_test_loader = DataLoader(internal_test_dataset, batch_size=batch_size, shuffle=False)
    
    # Also keep the original test loader for the competition's test data
    competition_test_loader = DataLoader(competition_test_dataset, batch_size=batch_size, shuffle=False)
    
    # Define model architecture
    model_type = "multimodal_gru"  # Options: "lstm", "gru", "transformer", "social_gru", "multimodal_gru"
    
    # Set model name for logging
    model_name = f"{model_type}_h{hidden_dim}_l{num_layers}_b{batch_size}_p{scale_position}"
    
    # Create model based on the selected type
    print(f"Creating {model_type} model...")
    
    if model_type == "lstm":
        model = Seq2SeqLSTMModel(
            input_dim=6,
            hidden_dim=hidden_dim,
            output_seq_len=60,
            output_dim=2,
            num_layers=num_layers
        )
    elif model_type == "gru":
        model = Seq2SeqGRUModel(
            input_dim=6,
            hidden_dim=hidden_dim,
            output_seq_len=60,
            output_dim=2,
            num_layers=num_layers
        )
    elif model_type == "transformer":
        model = Seq2SeqTransformerModel(
            input_dim=6,
            hidden_dim=hidden_dim,
            output_seq_len=60,
            output_dim=2,
            num_heads=4,
            num_layers=num_layers
        )
    elif model_type == "social_gru":
        model = SocialGRUModel(
            input_dim=6,
            hidden_dim=hidden_dim,
            output_seq_len=60,
            output_dim=2,
            num_layers=num_layers
        )
    elif model_type == "multimodal_gru":
        num_modes = 3  # Number of possible future trajectories to predict
        model = MultiModalGRUModel(
            input_dim=6,
            hidden_dim=hidden_dim,
            output_seq_len=60,
            output_dim=2,
            num_layers=num_layers,
            num_modes=num_modes
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Set device for training
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using CUDA")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    model = model.to(device)
    
    # Train model with appropriate training function
    print(f"Training {model_type} model...")
    
    if model_type == "multimodal_gru":
        model, checkpoint, metrics_history = train_multimodal_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=100,
            early_stopping_patience=10,
            lr=1e-3,
            weight_decay=1e-4,
            lr_step_size=20,
            lr_gamma=0.25,
            teacher_forcing_ratio=teacher_forcing_ratio,
            physics_weight=0.2,  # Control the influence of physics constraints
            model_name=model_name,
            save_logs=True
        )
    else:
        model, checkpoint, metrics_history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=100,
            early_stopping_patience=10,
            lr=1e-3,
            weight_decay=1e-4,
            lr_step_size=20,
            lr_gamma=0.25,
            teacher_forcing_ratio=teacher_forcing_ratio,
            model_name=model_name,
            save_logs=True
        )
    
    print(f"Best model saved with validation metrics:")
    print(f"  Normalized MSE: {checkpoint['val_loss']:.4f}")
    print(f"  MAE: {checkpoint['val_mae']:.4f}")
    print(f"  MSE: {checkpoint['val_mse']:.4f}")
    
    # Evaluate on internal test set
    print("Evaluating on internal test set...")
    
    # Use a different approach for internal test set evaluation
    model.eval()
    test_mse = 0.0
    test_mae = 0.0
    internal_test_predictions_list = []
    internal_test_gt_list = []
    internal_test_history_list = []
    
    with torch.no_grad():
        for batch in internal_test_loader:
            # Move data to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            # Get predictions - handle multi-modal case
            if hasattr(model, 'num_modes'):
                predictions, confidences = model(batch, teacher_forcing_ratio=0.0)
                # Get best prediction for each sample based on confidence
                best_modes = torch.argmax(confidences, dim=1)
                # Select best predictions
                best_predictions = torch.gather(
                    predictions, 
                    dim=2, 
                    index=best_modes.view(-1, 1, 1, 1).expand(-1, predictions.size(1), 1, predictions.size(-1))
                ).squeeze(2)  # [batch_size, seq_len, output_dim]
                predictions = best_predictions
            else:
                # For non-multimodal models
                predictions = model(batch, teacher_forcing_ratio=0.0)
            
            # Calculate unnormalized MSE and MAE
            # Unnormalize predictions
            pred_unnorm = predictions * batch['scale_position'].view(-1, 1, 1)
            # Unnormalize ground truth
            future_unnorm = batch['future'] * batch['scale_position'].view(-1, 1, 1)
            
            # Store for later visualization
            internal_test_predictions_list.append(pred_unnorm.cpu().numpy())
            internal_test_gt_list.append(future_unnorm.cpu().numpy())
            internal_test_history_list.append(batch['history'][:, 0, :, :2].cpu().numpy())
            
            # Calculate metrics
            test_mae += nn.L1Loss()(pred_unnorm, future_unnorm).item()
            test_mse += nn.MSELoss()(pred_unnorm, future_unnorm).item()
    
    # Calculate average metrics
    test_mae /= len(internal_test_loader)
    test_mse /= len(internal_test_loader)
    
    # Save test metrics
    test_metrics = {
        'model': model_name,
        'test_mae': test_mae,
        'test_mse': test_mse
    }
    
    # Append test metrics to a log file
    test_log_path = os.path.join("logs", "test_metrics.csv")
    import pandas as pd
    
    # Create DataFrame for test metrics
    test_df = pd.DataFrame([test_metrics])
    
    # Check if file exists to determine if header is needed
    if os.path.exists(test_log_path):
        test_df.to_csv(test_log_path, mode='a', header=False, index=False)
    else:
        test_df.to_csv(test_log_path, index=False)
    
    print(f"Internal test metrics:")
    print(f"  MAE: {test_mae:.4f}")
    print(f"  MSE: {test_mse:.4f}")
    print(f"Test metrics saved to {test_log_path}")
    
    # Visualize some predictions from internal test set
    # Combine predictions and ground truth from batches
    internal_test_predictions = np.concatenate(internal_test_predictions_list, axis=0)
    internal_test_gt = np.concatenate(internal_test_gt_list, axis=0)
    internal_test_history = np.concatenate(internal_test_history_list, axis=0)
    
    # Save visualization directory
    viz_dir = os.path.join("logs", f"{model_name}_test_viz")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Select a few samples for visualization
    num_viz_examples = 5
    indices = np.random.choice(len(internal_test_predictions), num_viz_examples, replace=False)
    
    print(f"Visualizing predictions for internal test set...")
    
    # Create combined visualization plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i in range(min(5, len(indices))):
        ax = axes[i]
        hist = internal_test_history[indices[i]]
        pred = internal_test_predictions[indices[i]]
        gt = internal_test_gt[indices[i]]
        
        ax.plot(hist[:, 0], hist[:, 1], 'ko-', label='History')
        ax.plot(pred[:, 0], pred[:, 1], 'bo-', label='Prediction')
        ax.plot(gt[:, 0], gt[:, 1], 'go-', label='Ground Truth')
        
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title(f'Example {i+1}')
        ax.grid(True)
        if i == 0:
            ax.legend()
    
    # Add test metrics to the last subplot
    if len(indices) < 6:
        ax = axes[5]
        ax.axis('off')
        metrics_text = (
            f"Test Metrics:\n"
            f"MAE: {test_mae:.4f}\n"
            f"MSE: {test_mse:.4f}\n\n"
            f"Best Val Metrics:\n"
            f"Val MAE: {checkpoint['val_mae']:.4f}\n"
            f"Val MSE: {checkpoint['val_mse']:.4f}\n"
        )
        ax.text(0.1, 0.5, metrics_text, fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "test_examples.png"), dpi=300)
    plt.close(fig)
    
    # Generate predictions for competition test data
    print("Generating predictions for competition test data...")
    if hasattr(model, 'num_modes'):
        competition_predictions = generate_multimodal_predictions(model, competition_test_loader)
    else:
        competition_predictions = generate_predictions(model, competition_test_loader)
    
    # Create submission file
    submission_path = f"{model_name}_submission.csv"
    print(f"Creating submission at {submission_path}...")
    create_submission(competition_predictions, output_file=submission_path)
    
    print("Done!")

if __name__ == "__main__":
    main()