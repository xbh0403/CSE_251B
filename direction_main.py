import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from data_utils.data_utils import TrajectoryDataset
from models.direction_model import DirectionAwareSeq2SeqGRUModel
from models.direction_train import train_direction_aware_model

def generate_predictions(model, test_loader):
    """Generate predictions for test data"""
    device = next(model.parameters()).device
    model.eval()
    
    all_predictions = []
    all_origins = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Generating predictions"):
            # Move data to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            # Get predictions (no teacher forcing for inference)
            predictions = model(batch, teacher_forcing_ratio=0.0)
            
            # Store predictions and origin for later denormalization
            all_predictions.append(predictions.cpu().numpy())
            all_origins.append(batch['origin'].cpu().numpy())
    
    # Concatenate all predictions and origins
    predictions = np.concatenate(all_predictions, axis=0)
    origins = np.concatenate(all_origins, axis=0)
    
    # Denormalize predictions
    scale_factor = test_loader.dataset.dataset.scale if hasattr(test_loader.dataset, 'dataset') else 7.0
    denormalized_predictions = predictions * scale_factor
    
    # Add origins - reshape origins to match predictions for broadcasting
    origins = origins.reshape(-1, 1, 2)  # [batch_size, 1, 2]
    denormalized_predictions = denormalized_predictions + origins
    
    return denormalized_predictions

def create_submission(predictions, output_file='submission.csv'):
    """Create submission file from predictions"""
    import pandas as pd
    
    # Check value ranges
    print(f"Prediction value ranges:")
    print(f"X min: {predictions[..., 0].min()}, X max: {predictions[..., 0].max()}")
    print(f"Y min: {predictions[..., 1].min()}, Y max: {predictions[..., 1].max()}")
    
    # Reshape predictions to match the expected format [batch_size * output_seq_len, 2]
    predictions_flat = predictions.reshape(-1, 2)
    
    # Create DataFrame
    submission_df = pd.DataFrame(predictions_flat, columns=['x', 'y'])
    
    # Add index column
    submission_df.index.name = 'index'
    
    # Save to CSV
    submission_df.to_csv(output_file)
    print(f'Submission saved to {output_file}')

def visualize_direction_samples(model, dataset, num_samples=5, save_dir='direction_samples'):
    """
    Visualize sample predictions to verify direction consistency
    """
    os.makedirs(save_dir, exist_ok=True)
    device = next(model.parameters()).device
    model.eval()
    
    # Create a small loader with the specified number of samples
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            # Get single sample
            sample = dataset[idx]
            
            # Convert to batch form and move to device
            batch = {}
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.unsqueeze(0).to(device)
            
            # Get prediction
            prediction = model(batch, teacher_forcing_ratio=0.0)
            
            # Get ground truth and history for visualization
            history = batch['history'][0, 0].cpu().numpy()  # [seq_len, feat_dim]
            if 'future' in batch:
                future = batch['future'][0].cpu().numpy()  # [seq_len, 2]
            else:
                future = None
                
            # Denormalize
            scale_factor = sample['scale'].item()
            origin = sample['origin'].numpy()
            
            # Denormalize history
            history_pos_denorm = history[:, :2] * scale_factor + origin
            
            # Denormalize prediction
            prediction_denorm = prediction[0].cpu().numpy() * scale_factor + origin
            
            # Denormalize future if available
            if future is not None:
                future_denorm = future * scale_factor + origin
            
            # Create visualization
            plt.figure(figsize=(10, 8))
            
            # Plot history
            plt.plot(history_pos_denorm[:, 0], history_pos_denorm[:, 1], 'b-', linewidth=2, label='History')
            plt.scatter(history_pos_denorm[0, 0], history_pos_denorm[0, 1], c='g', s=50, marker='o', label='Start')
            plt.scatter(history_pos_denorm[-1, 0], history_pos_denorm[-1, 1], c='b', s=50, marker='x', label='History End')
            
            # Plot prediction
            plt.plot(prediction_denorm[:, 0], prediction_denorm[:, 1], 'r-', linewidth=2, label='Prediction')
            plt.scatter(prediction_denorm[-1, 0], prediction_denorm[-1, 1], c='r', s=50, marker='x', label='Pred End')
            
            # Plot future if available
            if future is not None:
                plt.plot(future_denorm[:, 0], future_denorm[:, 1], 'g-', linewidth=2, label='Ground Truth')
                plt.scatter(future_denorm[-1, 0], future_denorm[-1, 1], c='g', s=50, marker='x', label='GT End')
            
            # Calculate and display direction vectors
            if len(history_pos_denorm) >= 2:
                # Last history direction
                last_hist_dir = history_pos_denorm[-1] - history_pos_denorm[-2]
                last_hist_dir = last_hist_dir / max(1e-6, np.linalg.norm(last_hist_dir))
                
                # First prediction direction
                first_pred_dir = prediction_denorm[0] - history_pos_denorm[-1]
                first_pred_dir = first_pred_dir / max(1e-6, np.linalg.norm(first_pred_dir))
                
                # Direction cosine similarity
                dir_cos_sim = np.dot(last_hist_dir, first_pred_dir)
                dir_angle = np.arccos(np.clip(dir_cos_sim, -1.0, 1.0)) * 180 / np.pi
                
                # Add direction vectors to plot
                arrow_scale = 2.0
                plt.arrow(history_pos_denorm[-1, 0], history_pos_denorm[-1, 1], 
                          last_hist_dir[0] * arrow_scale, last_hist_dir[1] * arrow_scale, 
                          head_width=0.5, head_length=0.7, fc='blue', ec='blue', label='History Direction')
                
                plt.arrow(history_pos_denorm[-1, 0], history_pos_denorm[-1, 1], 
                          first_pred_dir[0] * arrow_scale, first_pred_dir[1] * arrow_scale, 
                          head_width=0.5, head_length=0.7, fc='red', ec='red', label='Prediction Direction')
                
                plt.title(f"Direction Analysis - Angle: {dir_angle:.1f}°\nCosine Similarity: {dir_cos_sim:.3f}")
            else:
                plt.title("Direction Analysis - Insufficient history points")
            
            plt.xlabel('X Position (m)')
            plt.ylabel('Y Position (m)')
            plt.grid(True)
            plt.legend()
            plt.axis('equal')
            
            # Save the figure
            plt.savefig(os.path.join(save_dir, f'direction_sample_{i+1}.png'), dpi=200)
            plt.close()
    
    print(f"Direction samples saved to {save_dir}")

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Data paths (update these to your actual paths)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if "tscc" in current_dir:
        train_path = '/tscc/nfs/home/bax001/scratch/CSE_251B/data/train.npz'
        test_path = '/tscc/nfs/home/bax001/scratch/CSE_251B/data/test_input.npz'
    else:
        train_path = 'data/train.npz'
        test_path = 'data/test_input.npz'
    
    # Hyperparameters
    scale = 7.0
    batch_size = 64
    hidden_dim = 128
    num_layers = 2
    teacher_forcing_ratio = 0.6  # Increased for more stability
    
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
    print("Creating direction-aware model...")
    model = DirectionAwareSeq2SeqGRUModel(
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
    print("Training model with direction-aware losses...")
    model, checkpoint = train_direction_aware_model(
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
    print(f"  Total loss: {checkpoint['val_total_loss']:.4f}")
    print(f"  MSE loss: {checkpoint['val_mse_loss']:.4f}")
    print(f"  Direction loss: {checkpoint['val_direction_loss']:.4f}")
    print(f"  Initial direction loss: {checkpoint['val_initial_direction_loss']:.4f}")
    print(f"  MAE: {checkpoint['val_mae']:.4f}")
    
    # Visualize some samples to check direction consistency
    print("Generating direction sample visualizations...")
    visualize_direction_samples(model, val_dataset, num_samples=10, save_dir='direction_samples')
    
    # Generate predictions for test data
    print("Generating predictions...")
    predictions = generate_predictions(model, test_loader)
    
    # Create submission file
    print("Creating submission...")
    create_submission(predictions, output_file='direction_aware_submission.csv')
    
    print("Done!")

if __name__ == "__main__":
    main()