import sys
sys.path.append("/tscc/nfs/home/bax001/project/CSE_251B")
import os
os.chdir("/tscc/nfs/home/bax001/scratch/CSE_251B")

import torch
import numpy as np
from data_utils.data_utils import TrajectoryDataset
from models.model import GNNSequenceModel
from models.train import train_model
from models.predict import generate_predictions, create_submission
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Data paths
    train_path = '/tscc/nfs/home/bax001/scratch/CSE_251B/data/train.npz'
    test_path = '/tscc/nfs/home/bax001/scratch/CSE_251B/data/test_input.npz'
    
    # Load full training dataset
    print("Creating full training dataset...")
    full_train_dataset = TrajectoryDataset(train_path, split='train')
    
    test_dataset = TrajectoryDataset(test_path, split='test')
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    num_folds = 5
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    fold_results = []

    for fold_idx, (train_ids, val_ids) in enumerate(kfold.split(full_train_dataset)):
        print(f"--- Fold {fold_idx+1}/{num_folds} ---")

        # Create data loaders for the current fold
        train_subset = Subset(full_train_dataset, train_ids)
        val_subset = Subset(full_train_dataset, val_ids)

        train_loader = DataLoader(train_subset, batch_size=64, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_subset, batch_size=64, shuffle=False, num_workers=4)

        # Create model for the current fold
        print("Creating model for the current fold...")
        model = GNNSequenceModel(
            node_dim=6,
            gnn_hidden_dim=64,
            seq_hidden_dim=128,
            output_seq_len=60,
            output_dim=2,
            use_transformer=False
        )
        
        # Train model for the current fold
        print(f"Training model for fold {fold_idx+1}...")
        # Modify train_model to accept a fold_idx to save model specific to fold
        # And to return validation metrics
        trained_model, best_val_loss_fold = train_model(model, train_loader, val_loader, num_epochs=50, fold_idx=fold_idx+1)
        
        fold_results.append({'fold': fold_idx+1, 'val_loss': best_val_loss_fold})
        print(f"Fold {fold_idx+1} Best Validation Loss: {best_val_loss_fold:.4f}")

    print("\n--- Cross-Validation Summary ---")
    avg_val_loss = 0
    for result in fold_results:
        print(f"Fold {result['fold']}: Validation Loss = {result['val_loss']:.4f}")
        avg_val_loss += result['val_loss']
    avg_val_loss /= num_folds
    print(f"Average Validation Loss across {num_folds} folds: {avg_val_loss:.4f}")

    # For final predictions, you might want to load the best model from all folds
    # or train a new model on the full training data.
    # Here, as an example, we load the model from the first fold (or the one with the best val_loss).
    # Find the fold with the best validation loss
    best_fold_info = min(fold_results, key=lambda x: x['val_loss'])
    best_fold_idx = best_fold_info['fold']
    print(f"\nLoading model from best fold ({best_fold_idx}) for test set prediction...")
    
    # Re-create the model architecture
    final_model = GNNSequenceModel(
        node_dim=6,
        gnn_hidden_dim=64,
        seq_hidden_dim=128,
        output_seq_len=60,
        output_dim=2,
        use_transformer=False
    )
    model_path = f'best_model_lstm_fold_{best_fold_idx}.pth'
    final_model.load_state_dict(torch.load(model_path))
    
    # Generate predictions for test data
    print("Generating predictions...")
    predictions = generate_predictions(final_model, test_loader) # generate_predictions needs device handling for model
    
    # Create submission file
    print("Creating submission...")
    create_submission(predictions, output_file=f'gnn_lstm_submission_fold_{best_fold_idx}.csv')
    
    print("Done!")

if __name__ == "__main__":
    main()