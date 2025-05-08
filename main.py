import torch
import numpy as np
from data_utils.data_utils import TrajectoryDataset
from models.model import GNNSequenceModel
from models.train import train_model
from models.predict import generate_predictions, create_submission
from torch.utils.data import DataLoader, Subset

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Data paths
    train_path = '/tscc/nfs/home/bax001/scratch/CSE_251B/data/train.npz'
    test_path = '/tscc/nfs/home/bax001/scratch/CSE_251B/data/test_input.npz'
    
    # Create the full dataset
    print("Creating datasets...")
    full_train_dataset = TrajectoryDataset(train_path, split='train')
    test_dataset = TrajectoryDataset(test_path, split='test')
    
    # Create indices for train/validation split
    dataset_size = len(full_train_dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create subset datasets
    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_train_dataset, val_indices)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    
    # Create model
    print("Creating model...")
    model = GNNSequenceModel(
        node_dim=6,
        gnn_hidden_dim=64,
        seq_hidden_dim=128,
        output_seq_len=60,
        output_dim=2,
        use_transformer=True  # Set to False to use LSTM instead
    )
    
    # Train model - pass the original dataset for denormalization
    print("Training model...")
    model = train_model(model, train_loader, val_loader, full_train_dataset, full_train_dataset, num_epochs=50)
    
    # Load best model
    model.load_state_dict(torch.load('/tscc/nfs/home/bax001/scratch/CSE_251B/checkpoints/best_model.pth'))
    
    # Generate predictions for test data
    print("Generating predictions...")
    predictions = generate_predictions(model, test_loader, test_dataset)
    
    # Create submission file
    print("Creating submission...")
    create_submission(predictions, output_file='gnn_transformer_submission.csv')
    
    print("Done!")

if __name__ == "__main__":
    main()