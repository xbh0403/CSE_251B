import torch
import numpy as np
from data_utils.data_utils import TrajectoryDataset
from models.model import GNNSequenceModel
from models.train import train_model
from models.predict import generate_predictions, create_submission
from torch.utils.data import DataLoader

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Data paths
    train_path = 'data/train.npz'
    test_path = 'data/test_input.npz'
    
    # Data loaders
    print("Creating datasets...")
    train_val_dataset = TrajectoryDataset(train_path, split='train')
    
    # Split into train and validation
    train_size = int(0.8 * len(train_val_dataset))
    val_size = len(train_val_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_val_dataset, [train_size, val_size]
    )
    
    test_dataset = TrajectoryDataset(test_path, split='test')
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
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
    
    # Train model
    print("Training model...")
    model = train_model(model, train_loader, val_loader, num_epochs=30)
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    
    # Generate predictions for test data
    print("Generating predictions...")
    predictions = generate_predictions(model, test_loader)
    
    # Create submission file
    print("Creating submission...")
    create_submission(predictions, output_file='gnn_transformer_submission.csv')
    
    print("Done!")

if __name__ == "__main__":
    main()