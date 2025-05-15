import torch
import numpy as np
import os
from torch.utils.data import DataLoader, random_split
import torch.nn as nn

from data_utils.data_utils import TrajectoryDataset
from models.model import Seq2SeqLSTMModel, Seq2SeqGRUModel, Seq2SeqTransformerModel, SocialGRUModel
from models.train import train_model
from models.predict import generate_predictions, create_submission

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
    print("Creating model...")
    model = SocialGRUModel(
        input_dim=6,
        hidden_dim=hidden_dim,
        output_seq_len=60,
        output_dim=2,
        num_layers=num_layers
    )
    
    # Set device for training
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using CUDA")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    model = model.to(device)
    
    # Train model
    print("Training model...")
    model, checkpoint = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=100,
        early_stopping_patience=10,
        lr=1e-3,
        weight_decay=1e-4,
        lr_step_size=20,
        lr_gamma=0.25,
        teacher_forcing_ratio=teacher_forcing_ratio
    )
    
    print(f"Best model saved with validation metrics:")
    print(f"  Normalized MSE: {checkpoint['val_loss']:.4f}")
    print(f"  MAE: {checkpoint['val_mae']:.4f}")
    print(f"  MSE: {checkpoint['val_mse']:.4f}")
    
    # Evaluate on internal test set - directly calculating unnormalized MSE
    print("Evaluating on internal test set...")
    model.eval()
    test_mae = 0.0
    test_mse = 0.0
    
    with torch.no_grad():
        for batch in internal_test_loader:
            # Move data to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            # Forward pass (no teacher forcing during evaluation)
            predictions = model(batch, teacher_forcing_ratio=0.0)
            
            # Calculate unnormalized metrics
            pred_unnorm = predictions * batch['scale_position'].view(-1, 1, 1)
            future_unnorm = batch['future'] * batch['scale_position'].view(-1, 1, 1)
            
            test_mae += nn.L1Loss()(pred_unnorm, future_unnorm).item()
            test_mse += nn.MSELoss()(pred_unnorm, future_unnorm).item()
    
    # Calculate average test metrics
    test_mae /= len(internal_test_loader)
    test_mse /= len(internal_test_loader)
    
    print(f"Internal test metrics:")
    print(f"  MAE: {test_mae:.4f}")
    print(f"  MSE: {test_mse:.4f}")
    
    # Generate predictions for competition test data
    print("Generating predictions for competition test data...")
    competition_predictions = generate_predictions(model, competition_test_loader)
    
    # Create submission file
    print("Creating submission...")
    create_submission(competition_predictions, output_file='seq2seq_gru_submission.csv')
    
    print("Done!")

if __name__ == "__main__":
    main()