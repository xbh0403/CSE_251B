import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

def generate_predictions(model, test_loader):
    """Generate predictions for test data"""
    device = next(model.parameters()).device
    model.eval()
    
    all_predictions = []
    all_origins = []
    all_histories = []  # Also store histories to get exact last positions
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Generating predictions"):
            # Move data to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            # Get predictions (normalized) - no teacher forcing for inference
            predictions = model(batch, teacher_forcing_ratio=0.0)
            
            # Store predictions, history, and origin for later processing
            all_predictions.append(predictions.cpu().numpy())
            all_origins.append(batch['origin'].cpu().numpy())
            all_histories.append(batch['history'].cpu().numpy())
    
    # Concatenate all data
    predictions = np.concatenate(all_predictions, axis=0)
    origins = np.concatenate(all_origins, axis=0)
    histories = np.concatenate(all_histories, axis=0)
    
    # Denormalize predictions
    scale_factor_position = test_loader.dataset.dataset.scale_position if hasattr(test_loader.dataset, 'dataset') else 15.0
    scale_factor_velocity = test_loader.dataset.dataset.scale_velocity if hasattr(test_loader.dataset, 'dataset') else 5.0
    scale_factor_heading = test_loader.dataset.dataset.scale_heading if hasattr(test_loader.dataset, 'dataset') else 1.0
    denormalized_predictions = predictions * scale_factor_position
    
    # Add origins - reshape origins to match predictions for broadcasting
    origins = origins.reshape(-1, 1, 2)  # [batch_size, 1, 2]
    denormalized_predictions = denormalized_predictions + origins
    
    # IMPORTANT FIX: Ensure the first prediction exactly matches the last history point
    # Extract the last history point for each sample
    last_history_points = histories[:, 0, -1, :2]  # [batch_size, 2]
    
    # Denormalize these points
    denormalized_last_history = (last_history_points * scale_factor_position) + origins[:, 0, :]
    
    # Replace the first prediction with the exact last history point
    denormalized_predictions[:, 0, :] = denormalized_last_history
    
    return denormalized_predictions

def create_submission(predictions, output_file='submission.csv'):
    """Create submission file from predictions"""
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


def generate_multimodal_predictions(model, test_loader):
    """Generate predictions from a multi-modal model (using best confidence mode)"""
    device = next(model.parameters()).device
    model.eval()
    
    all_predictions = []
    all_origins = []
    all_histories = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Generating predictions"):
            # Move data to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            # Get predictions (all modes)
            if hasattr(model, 'num_modes'):
                predictions, confidences = model(batch, teacher_forcing_ratio=0.0)
                
                # Get best mode for each sample based on confidence
                best_modes = torch.argmax(confidences, dim=1)
                
                # Select best predictions
                best_predictions = torch.gather(
                    predictions, 
                    dim=2, 
                    index=best_modes.view(-1, 1, 1, 1).expand(-1, predictions.size(1), 1, predictions.size(-1))
                ).squeeze(2)  # [batch_size, seq_len, output_dim]
                
                # Use these as our final predictions
                predictions = best_predictions
            else:
                # For models without multi-modal support
                predictions = model(batch, teacher_forcing_ratio=0.0)
            
            # Store predictions, history, and origin for later processing
            all_predictions.append(predictions.cpu().numpy())
            all_origins.append(batch['origin'].cpu().numpy())
            all_histories.append(batch['history'].cpu().numpy())
    
    # Concatenate all data
    predictions = np.concatenate(all_predictions, axis=0)
    origins = np.concatenate(all_origins, axis=0)
    histories = np.concatenate(all_histories, axis=0)
    
    # Denormalize predictions
    scale_factor = test_loader.dataset.dataset.scale_position if hasattr(test_loader.dataset, 'dataset') else 15.0
    denormalized_predictions = predictions * scale_factor
    
    # Add origins - reshape origins to match predictions for broadcasting
    origins = origins.reshape(-1, 1, 2)  # [batch_size, 1, 2]
    denormalized_predictions = denormalized_predictions + origins
    
    # IMPORTANT FIX: Ensure the first prediction exactly matches the last history point
    # Extract the last history point for each sample
    last_history_points = histories[:, 0, -1, :2]  # [batch_size, 2]
    
    # Denormalize these points
    denormalized_last_history = (last_history_points * scale_factor) + origins[:, 0, :]
    
    # Replace the first prediction with the exact last history point
    denormalized_predictions[:, 0, :] = denormalized_last_history
    
    return denormalized_predictions