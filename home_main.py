# home_main.py

import torch
import numpy as np
import os
from torch.utils.data import DataLoader, random_split

from data_utils.data_utils import TrajectoryDataset
from models.home_model import HOMEModel
from models.home_train import train_home_model
from models.home_predict import generate_home_predictions, create_home_submission

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
    scale = 7.0  # From your data analysis
    batch_size = 64
    hidden_dim = 128
    num_layers = 2
    grid_size = 64  # Size of heatmap grid
    grid_range = [-50, 50]  # Range of heatmap in meters
    num_samples = 6  # Number of trajectory samples
    
    # Create the datasets
    print("Creating datasets...")
    full_train_dataset = TrajectoryDataset(train_path, split='train', scale=scale, augment=True)
    test_dataset = TrajectoryDataset(test_path, split='test', scale=scale)
    
    # Create train/validation split
    dataset_size = len(full_train_dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create HOME model
    print("Creating HOME model...")
    model = HOMEModel(
        input_dim=6,
        hidden_dim=hidden_dim,
        output_seq_len=60,
        grid_size=grid_size,
        grid_range=grid_range,
        num_samples=num_samples,
        num_layers=num_layers
    )
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using CUDA")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    model = model.to(device)
    
    # Train model
    print("Training HOME model...")
    model, checkpoint = train_home_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=100,
        early_stopping_patience=10,
        lr=1e-3,
        weight_decay=1e-4,
        lr_step_size=20,
        lr_gamma=0.25
    )
    
    print(f"Best model saved with validation metrics:")
    print(f"  Loss: {checkpoint['val_loss']:.4f}")
    print(f"  ADE: {checkpoint['val_ade']:.4f}")
    print(f"  FDE: {checkpoint['val_fde']:.4f}")
    
    # Generate predictions for test data
    print("Generating predictions...")
    predictions = generate_home_predictions(model, test_loader)
    
    # Create submission file
    print("Creating submission...")
    create_home_submission(predictions, output_file='home_submission.csv')
    
    print("Done!")

if __name__ == "__main__":
    main()