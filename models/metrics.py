import torch
import numpy as np
import matplotlib.pyplot as plt

def compute_metrics(predictions, ground_truth, scale_factor=None):
    """
    Compute trajectory prediction metrics
    
    Args:
        predictions: Predicted trajectories [batch_size, seq_len, 2]
        ground_truth: Ground truth trajectories [batch_size, seq_len, 2]
        scale_factor: Optional scale factor for denormalization
        
    Returns:
        Dictionary of metrics
    """
    # Apply scaling if provided
    if scale_factor is not None:
        if isinstance(predictions, torch.Tensor):
            scale = torch.tensor(scale_factor, device=predictions.device)
            predictions = predictions * scale.view(-1, 1, 1)
            ground_truth = ground_truth * scale.view(-1, 1, 1)
        else:
            predictions = predictions * scale_factor
            ground_truth = ground_truth * scale_factor
    
    # Calculate displacement error
    if isinstance(predictions, torch.Tensor):
        displacement_error = torch.norm(predictions - ground_truth, dim=2)
        ade = displacement_error.mean().item()
        fde = displacement_error[:, -1].mean().item()
        timestep_ade = displacement_error.mean(dim=0).cpu().numpy()
    else:
        displacement_error = np.linalg.norm(predictions - ground_truth, axis=2)
        ade = displacement_error.mean()
        fde = displacement_error[:, -1].mean()
        timestep_ade = displacement_error.mean(axis=0)
    
    return {
        'ADE': ade,
        'FDE': fde,
        'timestep_ade': timestep_ade
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