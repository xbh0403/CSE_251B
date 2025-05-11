import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# In models/predict.py

def generate_predictions(model, test_loader):
    """Generate predictions for test data with improved normalization"""
    device = next(model.parameters()).device
    model.eval()
    
    all_predictions = []
    all_origins = []
    all_position_scales = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Generating predictions"):
            # Move data to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            # Get predictions (normalized)
            predictions = model(batch)
            
            # Store predictions and normalization factors
            all_predictions.append(predictions.cpu().numpy())
            all_origins.append(batch['origin'].cpu().numpy())
            
            # Store position scale for denormalization
            all_position_scales.append(batch['position_scale'].cpu().numpy())
    
    # Concatenate all predictions and metadata
    predictions = np.concatenate(all_predictions, axis=0)
    origins = np.concatenate(all_origins, axis=0)
    position_scales = np.concatenate(all_position_scales, axis=0)
    
    # Denormalize predictions - using correct position scales
    position_scales = position_scales.reshape(-1, 1, 1)  # [batch_size, 1, 1]
    denormalized_predictions = predictions * position_scales
    
    # Add origins
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