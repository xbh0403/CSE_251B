import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

def generate_predictions(model, test_loader):
    """Generate predictions for test data"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    all_predictions = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader):
            # Move data to device
            history = batch['history'].to(device)
            
            # Prepare input data dictionary
            data = {'history': history}
            
            # Get predictions
            predictions = model(data)
            
            # Move predictions back to CPU
            predictions = predictions.cpu().numpy()
            
            # Collect predictions
            all_predictions.append(predictions)
    
    # Concatenate all predictions
    all_predictions = np.concatenate(all_predictions, axis=0)
    
    return all_predictions

def create_submission(predictions, output_file='submission.csv'):
    """Create submission file from predictions"""
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