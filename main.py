import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
import torch.nn as nn

from data_utils import TrajectoryDataset
from models.model import LSTMEncoder
from models.train import train_model
from models.predict import generate_predictions, create_submission

def find_optimal_epochs(train_path, scale=7.0, batch_size=128, hidden_dim=128, 
                       max_epochs=100, early_stopping_patience=10,
                       lr=1e-3, weight_decay=1e-4, lr_step_size=20, lr_gamma=0.25):
    """
    Find optimal number of training epochs using 70/15/15 split
    
    Args:
        train_path: Path to training data
        scale: Scaling factor for normalization
        batch_size: Batch size for training
        hidden_dim: Hidden dimension for LSTM
        max_epochs: Maximum number of epochs
        early_stopping_patience: Early stopping patience
        lr: Learning rate
        weight_decay: Weight decay
        lr_step_size: Learning rate scheduler step size
        lr_gamma: Learning rate scheduler gamma
        
    Returns:
        optimal_epochs: Number of epochs that achieved best validation performance
        best_checkpoint: Checkpoint with best validation metrics
    """
    print("=== PHASE 1: Finding Optimal Epochs ===")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create the full dataset
    print("Creating datasets...")
    full_train_dataset = TrajectoryDataset(train_path, split='train', scale=scale, augment=True)
    
    # Create train/validation/test split (70/15/15)
    dataset_size = len(full_train_dataset)
    train_size = int(0.7 * dataset_size)
    val_size = int(0.15 * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_train_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Dataset sizes:")
    print(f"  Training: {len(train_dataset)} samples ({len(train_dataset)/dataset_size*100:.1f}%)")
    print(f"  Validation: {len(val_dataset)} samples ({len(val_dataset)/dataset_size*100:.1f}%)")
    print(f"  Test (internal): {len(test_dataset)} samples ({len(test_dataset)/dataset_size*100:.1f}%)")
    
    # Create model
    print("Creating model...")
    model = LSTMEncoder(
        input_dim=6,
        hidden_dim=hidden_dim,
        output_dim=60*2
    )
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using CUDA")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    model = model.to(device)
    
    # Train model to find optimal epochs
    print("Training model to find optimal epochs...")
    model, checkpoint = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=max_epochs,
        early_stopping_patience=early_stopping_patience,
        lr=lr,
        weight_decay=weight_decay,
        lr_step_size=lr_step_size,
        lr_gamma=lr_gamma
    )
    
    optimal_epochs = checkpoint['epoch'] + 1  # +1 because epoch is 0-indexed
    
    print(f"Optimal training found:")
    print(f"  Best epoch: {optimal_epochs}")
    print(f"  Validation normalized MSE: {checkpoint['val_loss']:.4f}")
    print(f"  Validation MAE: {checkpoint['val_mae']:.4f}")
    print(f"  Validation MSE: {checkpoint['val_mse']:.4f}")
    
    # Evaluate on internal test set
    print("Evaluating on internal test set...")
    test_mse = evaluate_model_on_test(model, test_loader)
    print(f"Internal test unnormalized MSE: {test_mse:.4f}")
    
    return optimal_epochs, checkpoint

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
            
            # Extract ego agent history and get predictions
            history = batch['history']
            ego_history = history[:, 0, :, :]
            predictions = model(ego_history)
            
            # Denormalize predictions and ground truth
            pred_unnorm = predictions * batch['scale'].view(-1, 1, 1)
            future_unnorm = batch['future'] * batch['scale'].view(-1, 1, 1)
            
            # Calculate unnormalized MSE
            mse = nn.MSELoss()(pred_unnorm, future_unnorm)
            total_mse += mse.item()
            num_batches += 1
    
    return total_mse / num_batches

def train_on_full_dataset(train_path, optimal_epochs, scale=7.0, batch_size=128, hidden_dim=128,
                         lr=1e-3, weight_decay=1e-4, lr_step_size=20, lr_gamma=0.25):
    """
    Train model on full dataset for optimal number of epochs
    
    Args:
        train_path: Path to training data
        optimal_epochs: Number of epochs to train for
        scale: Scaling factor for normalization
        batch_size: Batch size for training
        hidden_dim: Hidden dimension for LSTM
        lr: Learning rate
        weight_decay: Weight decay
        lr_step_size: Learning rate scheduler step size
        lr_gamma: Learning rate scheduler gamma
        
    Returns:
        trained_model: Model trained on full dataset
    """
    print("\n=== PHASE 2: Training on Full Dataset ===")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create the full dataset (no split this time)
    print("Creating full training dataset...")
    full_train_dataset = TrajectoryDataset(train_path, split='train', scale=scale, augment=True)
    
    # Create data loader for full dataset
    full_train_loader = DataLoader(full_train_dataset, batch_size=batch_size, shuffle=True)
    
    print(f"Full training dataset size: {len(full_train_dataset)} samples")
    
    # Create fresh model
    print("Creating fresh model...")
    model = LSTMEncoder(
        input_dim=6,
        hidden_dim=hidden_dim,
        output_dim=60*2
    )
    
    # Set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    
    # Setup training
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)
    
    print(f"Training on full dataset for {optimal_epochs} epochs...")
    
    # Training loop for exact number of epochs
    from tqdm import tqdm
    progress_bar = tqdm(range(optimal_epochs), desc="Epoch", unit="epoch")
    
    for epoch in progress_bar:
        model.train()
        train_loss = 0.0
        train_mse_unnorm = 0.0
        
        for batch in full_train_loader:
            # Move data to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            history = batch['history']
            ego_history = history[:, 0, :, :]
            predictions = model(ego_history)
            
            # Calculate loss (normalized for backprop)
            loss = criterion(predictions, batch['future'])
            
            # Calculate unnormalized MSE for reporting
            pred_unnorm = predictions * batch['scale'].view(-1, 1, 1)
            future_unnorm = batch['future'] * batch['scale'].view(-1, 1, 1)
            unnorm_mse = nn.MSELoss()(pred_unnorm, future_unnorm)
            
            # Backward and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_mse_unnorm += unnorm_mse.item()
        
        # Calculate average training losses
        train_loss /= len(full_train_loader)
        train_mse_unnorm /= len(full_train_loader)
        
        # Update learning rate
        scheduler.step()
        
        # Update progress bar
        progress_bar.set_postfix({
            'lr': f"{optimizer.param_groups[0]['lr']:.6f}",
            'train_mse': f"{train_mse_unnorm:.4f}"
        })
    
    # Save final model
    torch.save({
        'epoch': optimal_epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_mse': train_mse_unnorm
    }, "final_model.pth")
    
    print(f"Final model trained and saved!")
    print(f"Final training unnormalized MSE: {train_mse_unnorm:.4f}")
    
    return model

def main():
    """Main function implementing the optimal training strategy"""
    # Data paths
    train_path = '/tscc/nfs/home/bax001/scratch/CSE_251B/data/train.npz'
    test_path = '/tscc/nfs/home/bax001/scratch/CSE_251B/data/test_input.npz'
    
    # Hyperparameters
    scale = 7.0
    batch_size = 128
    hidden_dim = 128
    
    # Phase 1: Find optimal epochs using 3-way split
    optimal_epochs, best_checkpoint = find_optimal_epochs(
        train_path=train_path,
        scale=scale,
        batch_size=batch_size,
        hidden_dim=hidden_dim,
        max_epochs=100,
        early_stopping_patience=10,
        lr=1e-3,
        weight_decay=1e-4,
        lr_step_size=20,
        lr_gamma=0.25
    )
    
    # Phase 2: Train on full dataset for optimal epochs
    final_model = train_on_full_dataset(
        train_path=train_path,
        optimal_epochs=optimal_epochs,
        scale=scale,
        batch_size=batch_size,
        hidden_dim=hidden_dim,
        lr=1e-3,
        weight_decay=1e-4,
        lr_step_size=20,
        lr_gamma=0.25
    )
    
    # Phase 3: Generate predictions for external test set
    print("\n=== PHASE 3: Generating Final Predictions ===")
    external_test_dataset = TrajectoryDataset(test_path, split='test', scale=scale)
    external_test_loader = DataLoader(external_test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"External test dataset size: {len(external_test_dataset)} samples")
    print("Generating predictions for external test set...")
    
    final_predictions = generate_predictions(final_model, external_test_loader)
    create_submission(final_predictions, output_file='optimal_lstm_submission.csv')
    
    print("\n=== SUMMARY ===")
    print(f"Optimal epochs found: {optimal_epochs}")
    print(f"Best validation MSE: {best_checkpoint['val_mse']:.4f}")
    print(f"Final model trained on full dataset")
    print(f"Predictions saved to: optimal_lstm_submission.csv")
    print("Done!")

if __name__ == "__main__":
    main() 