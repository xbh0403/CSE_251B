import torch
import numpy as np
import os
from torch.utils.data import DataLoader, random_split
import torch.nn as nn

from data_utils.data_utils import TrajectoryDataset
from models.model import Seq2SeqLSTMModel, Seq2SeqGRUModel, Seq2SeqTransformerModel, SocialGRUModel, MultiModalGRUModel
from models.train import train_model, train_multimodal_model
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
    
    # Create model
    # Create model - Use the multi-modal GRU model
    print("Creating model...")
    num_modes = 3  # Number of possible future trajectories to predict
    model = MultiModalGRUModel(
        input_dim=6,
        hidden_dim=hidden_dim,
        output_seq_len=60,
        output_dim=2,
        num_layers=num_layers,
        num_modes=num_modes
    )
    
    # Set device for training
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using CUDA")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    model = model.to(device)
    
    # Train model with the multi-modal training function
    print("Training model...")
    # In main.py, add physics_weight parameter to the train_multimodal_model call
    model, checkpoint = train_multimodal_model(
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
        physics_weight=0.2  # Control the influence of physics constraints
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
    
    print(f"Internal test metrics:")
    print(f"  MAE: {test_mae:.4f}")
    print(f"  MSE: {test_mse:.4f}")
    
    # Visualize some predictions from internal test set
    # Combine predictions and ground truth from batches
    internal_test_predictions = np.concatenate(internal_test_predictions_list, axis=0)
    internal_test_gt = np.concatenate(internal_test_gt_list, axis=0)
    internal_test_history = np.concatenate(internal_test_history_list, axis=0)
    
    # Select a few samples for visualization
    num_viz_examples = 5
    indices = np.random.choice(len(internal_test_predictions), num_viz_examples, replace=False)
    
    print("Visualizing predictions for internal test set...")
    visualize_predictions(
        internal_test_history[indices],
        internal_test_predictions[indices],
        internal_test_gt[indices],
        num_examples=num_viz_examples
    )
    
    # Generate predictions for competition test data...
    print("Generating predictions for competition test data...")
    competition_predictions = generate_multimodal_predictions(model, competition_test_loader)
    
    # Create submission file
    print("Creating submission...")
    create_submission(competition_predictions, output_file='multimodal_gru_submission.csv')
    
    print("Done!")

if __name__ == "__main__":
    main()