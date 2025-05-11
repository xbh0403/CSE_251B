import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split

# Import your existing modules
from data_utils.data_utils import TrajectoryDataset
from models.predict import generate_predictions, create_submission
from models.metrics import compute_metrics, visualize_predictions

# Import the PhysicsGuidedLSTM class - place it in models/model.py
from models.model import PhysicsGuidedLSTM

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Data paths - use your existing paths
    train_path = '/tscc/nfs/home/bax001/scratch/CSE_251B/data/train.npz'
    test_path = '/tscc/nfs/home/bax001/scratch/CSE_251B/data/test_input.npz'
    
    # Hyperparameters - keep your existing hyperparameters
    scale = 7.0  # Using the scale that works well
    batch_size = 32
    hidden_dim = 128
    
    # Create datasets with your existing TrajectoryDataset class
    print("Creating datasets...")
    full_train_dataset = TrajectoryDataset(train_path, split='train', scale=scale, augment=True)
    test_dataset = TrajectoryDataset(test_path, split='test', scale=scale)
    
    # Create train/validation/test split (80%, 10%, 10%)
    dataset_size = len(full_train_dataset)
    train_size = int(0.8 * dataset_size)
    val_size = int(0.1 * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    # Create indices for the splits
    indices = list(range(dataset_size))
    train_indices, temp_indices = train_test_split(
        indices, test_size=(val_size + test_size)/dataset_size, random_state=42
    )
    val_indices, test_indices = train_test_split(
        temp_indices, test_size=test_size/(val_size + test_size), random_state=42
    )
    
    # Create subset datasets
    from torch.utils.data import Subset
    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_train_dataset, val_indices)
    test_from_train_dataset = Subset(full_train_dataset, test_indices)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_from_train_loader = DataLoader(test_from_train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Check input dimensions from an actual batch
    sample_batch = next(iter(train_loader))
    history_shape = sample_batch['history'].shape
    print(f"History shape: {history_shape}")
    
    # Get input dimension for the model
    if len(history_shape) == 4:  # [batch, num_agents, seq_len, feat_dim]
        input_dim = history_shape[-1]
    else:  # [batch, seq_len, feat_dim]
        input_dim = history_shape[-1]
    
    print(f"Input dimension: {input_dim}")
    
    # Create the physics-guided LSTM model
    print("Creating physics-guided model...")
    model = PhysicsGuidedLSTM(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_seq_len=60,
        output_dim=2
    )
    
    # Set device for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)
    
    # Train model using your existing train function
    from models.train import train_model
    
    print("Training physics-guided model...")
    model, checkpoint = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=50,  # Reduced for initial testing
        early_stopping_patience=10,
        lr=1e-3,
        weight_decay=1e-4,
        lr_step_size=15,
        lr_gamma=0.5
    )
    
    print(f"Best model saved with validation metrics:")
    print(f"  Normalized MSE: {checkpoint['val_loss']:.4f}")
    print(f"  MAE: {checkpoint['val_mae']:.4f}")
    print(f"  MSE: {checkpoint['val_mse']:.4f}")
    
    # Evaluate on test split from training data
    print("Evaluating model on test split...")
    model.eval()
    with torch.no_grad():
        all_predictions = []
        all_ground_truth = []
        all_origins = []
        all_scales = []
        
        for batch in test_from_train_loader:
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            # Forward pass
            predictions = model(batch)
            
            # Store predictions and ground truth
            all_predictions.append(predictions.cpu().numpy())
            all_ground_truth.append(batch['future'].cpu().numpy())
            all_origins.append(batch['origin'].cpu().numpy())
            all_scales.append(batch['scale'].cpu().numpy())
        
        # Concatenate results
        predictions = np.concatenate(all_predictions, axis=0)
        ground_truth = np.concatenate(all_ground_truth, axis=0)
        origins = np.concatenate(all_origins, axis=0)
        scales = np.concatenate(all_scales, axis=0)
        
        # Denormalize predictions and ground truth
        scales = scales.reshape(-1, 1, 1)
        origins = origins.reshape(-1, 1, 2)
        
        denorm_predictions = predictions * scales + origins
        denorm_ground_truth = ground_truth * scales + origins
        
        # Calculate metrics
        metrics = compute_metrics(denorm_predictions, denorm_ground_truth)
        
        print(f"Test metrics:")
        print(f"  MAE: {metrics['MAE']:.4f}")
        print(f"  MSE: {metrics['MSE']:.4f}")
    
    # Generate predictions for final test data
    print("Generating predictions for test data...")
    predictions = generate_predictions(model, test_loader)
    
    # Create submission file
    print("Creating submission...")
    create_submission(predictions, output_file='physics_lstm_basic_submission.csv')
    
    print("Done!")

if __name__ == "__main__":
    main()