import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
import torch.nn as nn

from data_utils import TrajectoryDataset
from models.model import LSTMModel
from models.train import train_model
from models.predict import generate_predictions, create_submission

def evaluate_model_on_test(model, test_loader):
    """Evaluate the model on test data and return unnormalized MSE"""
    device = next(model.parameters()).device
    model.eval()
    
    total_mse = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in test_loader:
            # Move data to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            # Get predictions (normalized)
            predictions = model(batch)
            
            # Denormalize predictions and ground truth
            pred_unnorm = predictions * batch['scale'].view(-1, 1, 1)
            future_unnorm = batch['future'] * batch['scale'].view(-1, 1, 1)
            
            # Calculate unnormalized MSE
            mse = nn.MSELoss()(pred_unnorm, future_unnorm)
            total_mse += mse.item()
            num_batches += 1
    
    return total_mse / num_batches

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Data paths (update these to your actual paths)
    train_path = '/tscc/nfs/home/bax001/scratch/CSE_251B/data/train.npz'
    test_path = '/tscc/nfs/home/bax001/scratch/CSE_251B/data/test_input.npz'
    
    # Hyperparameters
    scale = 7.0
    batch_size = 128
    hidden_dim = 128
    
    # Create the full dataset
    print("Creating datasets...")
    full_train_dataset = TrajectoryDataset(train_path, split='train', scale=scale, augment=True)
    test_dataset = TrajectoryDataset(test_path, split='test', scale=scale)
    
    # Create train/validation/test split (70/15/15)
    dataset_size = len(full_train_dataset)
    train_size = int(0.7 * dataset_size)
    val_size = int(0.15 * dataset_size)
    test_size = dataset_size - train_size - val_size  # Remaining data for test
    
    train_dataset, val_dataset, test_dataset_from_train = random_split(
        full_train_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset_from_train, batch_size=batch_size, shuffle=False)
    
    # Optional: Keep the original test dataset from file if needed
    original_test_dataset = TrajectoryDataset(test_path, split='test', scale=scale)
    original_test_loader = DataLoader(original_test_dataset, batch_size=batch_size, shuffle=False)
    
    # Print dataset sizes
    print(f"Dataset sizes:")
    print(f"  Training: {len(train_dataset)} samples ({len(train_dataset)/dataset_size*100:.1f}%)")
    print(f"  Validation: {len(val_dataset)} samples ({len(val_dataset)/dataset_size*100:.1f}%)")
    print(f"  Test (from train data): {len(test_dataset_from_train)} samples ({len(test_dataset_from_train)/dataset_size*100:.1f}%)")
    print(f"  Original test file: {len(original_test_dataset)} samples")
    
    # Create model
    print("Creating model...")
    model = LSTMModel(
        input_dim=6,
        hidden_dim=hidden_dim,
        output_seq_len=60,
        output_dim=2
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
        lr_gamma=0.25
    )
    
    print(f"Best model saved with validation metrics:")
    print(f"  Normalized MSE: {checkpoint['val_loss']:.4f}")
    print(f"  MAE: {checkpoint['val_mae']:.4f}")
    print(f"  MSE: {checkpoint['val_mse']:.4f}")
    
    # Evaluate on internal test set
    print("Evaluating model on internal test set...")
    test_mse = evaluate_model_on_test(model, test_loader)
    print(f"Test set unnormalized MSE: {test_mse:.4f}")
    
    # Generate predictions for original test file (for submission)
    print("Generating predictions for original test file...")
    original_predictions = generate_predictions(model, original_test_loader)
    create_submission(original_predictions, output_file='lstm_submission.csv')
    
    print("Done!")

if __name__ == "__main__":
    main()