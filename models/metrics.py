import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F # Import F for functional losses

def compute_metrics(predictions, ground_truth, scale_factor=None):
    """
    Compute trajectory prediction metrics (MAE and MSE)
    
    Args:
        predictions: Predicted trajectories [batch_size, seq_len, 2]
        ground_truth: Ground truth trajectories [batch_size, seq_len, 2]
        scale_factor: Optional scale factor for denormalization
        
    Returns:
        Dictionary of metrics {'MAE': mae_value, 'MSE': mse_value}
    """
    # Apply scaling if provided
    if scale_factor is not None:
        if isinstance(predictions, torch.Tensor):
            # Ensure scale_factor is a tensor with the same device and dtype
            if not isinstance(scale_factor, torch.Tensor):
                scale = torch.tensor(scale_factor, device=predictions.device, dtype=predictions.dtype)
            else:
                scale = scale_factor.to(device=predictions.device, dtype=predictions.dtype)
            
            # Ensure scale can be broadcasted; if it's scalar, it should apply to both x and y
            if scale.ndim == 0 or (scale.ndim == 1 and scale.size(0) == 1) : # scalar scale
                 predictions = predictions * scale
                 ground_truth = ground_truth * scale
            elif scale.ndim == 1 and scale.size(0) == predictions.size(2): # per-coordinate scale e.g. [scale_x, scale_y]
                 predictions = predictions * scale.view(1, 1, -1) 
                 ground_truth = ground_truth * scale.view(1, 1, -1)
            elif scale.ndim == 3 and scale.shape == predictions.shape: # full scale tensor
                 predictions = predictions * scale
                 ground_truth = ground_truth * scale
            else: # per-batch item scale
                 predictions = predictions * scale.view(-1, 1, 1)
                 ground_truth = ground_truth * scale.view(-1, 1, 1)

        else: # Numpy arrays
            predictions = predictions * scale_factor
            ground_truth = ground_truth * scale_factor
    
    if isinstance(predictions, torch.Tensor):
        mae = F.l1_loss(predictions, ground_truth).item()
        mse = F.mse_loss(predictions, ground_truth).item()
    else: # Assuming numpy array
        mae = np.mean(np.abs(predictions - ground_truth))
        mse = np.mean((predictions - ground_truth)**2)
    
    return {
        'MAE': mae,
        'MSE': mse
    }

def visualize_predictions(history, predictions, ground_truth=None, num_examples=5):
    """
    Visualize trajectory predictions
    
    Args:
        history: Historical trajectories [batch_size, seq_len, 2]
        predictions: Predicted trajectories [batch_size, seq_len, 2]
        ground_truth: Optional ground truth trajectories [batch_size, seq_len, 2]
        num_examples: Number of examples to visualize
    """
    # Select random examples if there are more than num_examples
    batch_size = min(len(predictions), num_examples)
    indices = np.random.choice(len(predictions), batch_size, replace=False)
    
    for i, idx in enumerate(indices):
        plt.figure(figsize=(10, 8))
        
        # Plot historical trajectory
        hist = history[idx]
        plt.plot(hist[:, 0], hist[:, 1], 'ko-', label='History')
        
        # Plot prediction
        pred = predictions[idx]
        plt.plot(pred[:, 0], pred[:, 1], 'bo-', label='Prediction')
        
        # Plot ground truth if available
        if ground_truth is not None:
            gt = ground_truth[idx]
            plt.plot(gt[:, 0], gt[:, 1], 'go-', label='Ground Truth')
        
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title(f'Trajectory Example {i+1}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'trajectory_example_{i+1}.png')
        plt.close()