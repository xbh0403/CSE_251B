import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

def generate_predictions(model, test_loader, test_dataset):
    """Generate predictions for test data with denormalization"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    all_predictions = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader):
            # Handle different batch structures
            if isinstance(batch, dict):
                history = batch['history'].to(device)
            else:
                # If your Subset returns a different structure, handle accordingly
                history = batch[0].to(device)  # Adjust as needed
            
            # Prepare input data dictionary
            data = {'history': history}
            
            # Calculate baseline predictions
            from data_utils.feature_engineering import compute_constant_velocity
            baseline_pred = compute_constant_velocity(history)
            
            # Get model predictions (as residuals)
            residuals = model(data)
            
            # Apply residuals to baseline
            predictions = baseline_pred + residuals
            
            # Denormalize predictions before saving
            denormalized_predictions = test_dataset.denormalize_positions(predictions)
            
            # Move predictions back to CPU
            denormalized_predictions = denormalized_predictions.cpu().numpy()
            
            # Collect predictions
            all_predictions.append(denormalized_predictions)
    
    # Concatenate all predictions
    all_predictions = np.concatenate(all_predictions, axis=0)
    
    return all_predictions

def create_submission(predictions, output_file='submission.csv'):
    """Create submission file from predictions with value checks"""
    # Check value ranges
    print(f"Prediction value ranges:")
    print(f"X min: {predictions[..., 0].min()}, X max: {predictions[..., 0].max()}")
    print(f"Y min: {predictions[..., 1].min()}, Y max: {predictions[..., 1].max()}")
    
    # If these values are very small (like between -3 and 3), they're likely still normalized!
    if abs(predictions[..., 0].max() - predictions[..., 0].min()) < 10:
        print("WARNING: Predictions appear to still be normalized! Check denormalization.")
    
    # Reshape predictions to match the expected format
    # From [batch_size, output_seq_len, 2] to [batch_size * output_seq_len, 2]
    predictions_flat = predictions.reshape(-1, 2)
    
    # Create DataFrame
    submission_df = pd.DataFrame(predictions_flat, columns=['x', 'y'])
    
    # Add index column
    submission_df.index.name = 'index'
    
    # Save to CSV
    submission_df.to_csv(output_file)
    print(f'Submission saved to {output_file}')