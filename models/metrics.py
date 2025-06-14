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
        Dictionary of metrics containing the following keys:
            'MAE' → Mean Absolute Error
            'MSE' → Mean Squared Error
            'ADE' → Average Displacement Error
            'FDE' → Final Displacement Error
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
    
    # Compute MAE and MSE using PyTorch or NumPy operations depending on the input type
    if isinstance(predictions, torch.Tensor):
        mae = F.l1_loss(predictions, ground_truth).item()
        mse = F.mse_loss(predictions, ground_truth).item()

        # Convert to numpy for ADE/FDE calculation (safer for broadcasting and to avoid additional GPU ops)
        preds_np = predictions.detach().cpu().numpy()
        gt_np = ground_truth.detach().cpu().numpy()
    else:  # NumPy arrays
        preds_np = predictions
        gt_np = ground_truth
        mae = np.mean(np.abs(preds_np - gt_np))
        mse = np.mean((preds_np - gt_np) ** 2)

    # Average Displacement Error (ADE) and Final Displacement Error (FDE)
    # ADE: Mean Euclidean distance over all timesteps and samples
    # FDE: Mean Euclidean distance at the final timestep
    displacement = np.linalg.norm(preds_np - gt_np, axis=2)  # Shape: [batch, seq_len]
    ade = np.mean(displacement)
    fde = np.mean(displacement[:, -1])

    return {
        'MAE': mae,
        'MSE': mse,
        'ADE': ade,
        'FDE': fde
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