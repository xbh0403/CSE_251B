# models/home_predict.py

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

def generate_home_predictions(model, test_loader):
    """
    Generate predictions using HOME model
    
    Args:
        model: HOME model
        test_loader: DataLoader for test data
        
    Returns:
        denormalized_predictions: [num_test_samples, output_seq_len, 2]
    """
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
            
            # Get object types
            obj_types = batch['history'][:, 0, 0, 5].long()
            obj_types = torch.clamp(obj_types, min=0, max=9)
            
            # Generate multimodal predictions
            predictions, confidences = model(batch, teacher_forcing_ratio=0.0)
            
            # Predictions shape: [batch_size, num_samples, output_seq_len, 2]
            # Confidences shape: [batch_size, num_samples]
            
            # Select most confident prediction for each sample
            best_idx = torch.argmax(confidences, dim=1)
            batch_size = predictions.shape[0]
            
            selected_predictions = torch.zeros(batch_size, model.output_seq_len, 2, device=device)
            for i in range(batch_size):
                selected_predictions[i] = predictions[i, best_idx[i]]
            
            # Store predictions and origin for later denormalization
            all_predictions.append(selected_predictions.cpu().numpy())
            all_origins.append(batch['origin'].cpu().numpy())
    
    # Concatenate all data
    predictions = np.concatenate(all_predictions, axis=0)
    origins = np.concatenate(all_origins, axis=0)
    
    # Denormalize predictions
    scale_factor = test_loader.dataset.dataset.scale if hasattr(test_loader.dataset, 'dataset') else 7.0
    denormalized_predictions = predictions * scale_factor
    
    # Add origins - reshape origins to match predictions
    origins = origins.reshape(-1, 1, 2)
    denormalized_predictions = denormalized_predictions + origins
    
    # IMPORTANT: Ensure the first prediction smoothly continues from history
    # Extract the last history point for each sample (already part of origin)
    last_history_points = origins[:, 0, :]
    
    # Add a slight adjustment to first predicted point to ensure continuity
    first_pred_dir = denormalized_predictions[:, 0, :] - last_history_points
    first_pred_dist = np.linalg.norm(first_pred_dir, axis=1, keepdims=True)
    first_pred_dir = first_pred_dir / (first_pred_dist + 1e-6)
    
    # Set first point to be very close to last history point (smoothing)
    denormalized_predictions[:, 0, :] = last_history_points + first_pred_dir * 0.1
    
    return denormalized_predictions

def create_home_submission(predictions, output_file='home_submission.csv'):
    """
    Create submission file from HOME predictions
    
    Args:
        predictions: Trajectory predictions [num_test_samples, output_seq_len, 2]
        output_file: Output CSV file path
    """
    # Check value ranges
    print(f"Prediction value ranges:")
    print(f"X min: {predictions[..., 0].min()}, X max: {predictions[..., 0].max()}")
    print(f"Y min: {predictions[..., 1].min()}, Y max: {predictions[..., 1].max()}")
    
    # Reshape predictions to match expected format [num_test_samples * output_seq_len, 2]
    predictions_flat = predictions.reshape(-1, 2)
    
    # Create DataFrame
    submission_df = pd.DataFrame(predictions_flat, columns=['x', 'y'])
    
    # Add index column
    submission_df.index.name = 'index'
    
    # Save to CSV
    submission_df.to_csv(output_file)
    print(f'Submission saved to {output_file}')