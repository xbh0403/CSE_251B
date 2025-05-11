import torch
import numpy as np
from torch.utils.data import DataLoader, random_split

from data_utils.data_utils import TrajectoryDataset
from models.model import LSTMModel, GRUModel, TransformerModel
from models.train import train_model
from models.predict import generate_predictions, create_submission
from models.metrics import compute_metrics, visualize_predictions

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Data paths (update these to your actual paths)
    train_path = '/tscc/nfs/home/bax001/scratch/CSE_251B/data/train.npz'
    test_path = '/tscc/nfs/home/bax001/scratch/CSE_251B/data/test_input.npz'
    
    # Hyperparameters
    scale = 7.0
    batch_size = 64
    hidden_dim = 128
    
    # Create the full dataset
    print("Creating datasets...")
    full_train_dataset = TrajectoryDataset(train_path, split='train', scale=scale, augment=True)
    test_dataset = TrajectoryDataset(test_path, split='test', scale=scale)
    
    # Create train/validation split
    dataset_size = len(full_train_dataset)
    train_size = int(0.9 * dataset_size)
    val_size = dataset_size - train_size
    
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    print("Creating model...")
    model = GRUModel(
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
    
    # Generate predictions for test data
    print("Generating predictions...")
    predictions = generate_predictions(model, test_loader)
    
    # Create submission file
    print("Creating submission...")
    create_submission(predictions, output_file='lstm_submission.csv')
    
    print("Done!")

if __name__ == "__main__":
    main()