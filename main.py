import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split

from data_utils.data_utils import TrajectoryDataset
from models.model import LSTMModel
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
    position_scale = 7.0  # Maintain the scale that works well
    velocity_scale = 5.0  # Separate scale for velocities
    batch_size = 32
    hidden_dim = 128
    robust_norm = True  # Enable robust normalization
    
    # Create the full dataset
    print("Creating datasets...")
    full_train_dataset = TrajectoryDataset(
        train_path, 
        split='train', 
        position_scale=position_scale,
        velocity_scale=velocity_scale,
        robust_norm=robust_norm,
        augment=True
    )
    test_dataset = TrajectoryDataset(
        test_path, 
        split='test', 
        position_scale=position_scale,
        velocity_scale=velocity_scale,
        robust_norm=robust_norm
    )
    
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
    
    # Create model with adjusted input dimension if needed
    print("Creating model...")
    input_dim = 6  # Default input dimension
    
    model = LSTMModel(
        input_dim=input_dim,
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
    
    # Train model with slightly adjusted parameters
    print("Training model...")
    model, checkpoint = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=100,
        early_stopping_patience=15,  # Increase patience slightly
        lr=1e-3,
        weight_decay=1e-4,
        lr_step_size=15,  # Adjust learning rate schedule
        lr_gamma=0.5  # Less aggressive reduction
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
    print("Generating predictions...")
    predictions = generate_predictions(model, test_loader)
    
    # Create submission file
    print("Creating submission...")
    create_submission(predictions, output_file='improved_submission.csv')
    
    print("Done!")

if __name__ == "__main__":
    main()