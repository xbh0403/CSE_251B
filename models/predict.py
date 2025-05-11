import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

def generate_predictions(model, test_loader):
    """Generate predictions for test data with improved normalization handling"""
    device = next(model.parameters()).device
    model.eval()
    
    all_predictions = []
    all_origins = []
    all_scales = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Generating predictions"):
            # Move data to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            # Get predictions (normalized)
            predictions = model(batch)
            
            # Store predictions and origin for later denormalization
            all_predictions.append(predictions.cpu().numpy())
            all_origins.append(batch['origin'].cpu().numpy())
            all_scales.append(batch['scale'].cpu().numpy())
    
    # Concatenate all predictions and origins
    predictions = np.concatenate(all_predictions, axis=0)
    origins = np.concatenate(all_origins, axis=0)
    scales = np.concatenate(all_scales, axis=0)
    
    # Denormalize predictions - using individual scales for each trajectory
    # but converting to proper shape for broadcasting
    scales = scales.reshape(-1, 1, 1)  # [batch_size, 1, 1]
    denormalized_predictions = predictions * scales
    
    # Add origins - reshape origins to match predictions for broadcasting
    origins = origins.reshape(-1, 1, 2)  # [batch_size, 1, 2]
    denormalized_predictions = denormalized_predictions + origins
    
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