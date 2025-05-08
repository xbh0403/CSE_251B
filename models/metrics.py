import torch

def compute_metrics(predictions, ground_truth):
    """
    Compute trajectory prediction metrics
    
    Args:
        predictions: Predicted trajectories [batch_size, seq_len, 2]
        ground_truth: Ground truth trajectories [batch_size, seq_len, 2]
        
    Returns:
        Dictionary of metrics:
            - ADE: Average Displacement Error (across all timesteps)
            - FDE: Final Displacement Error (at the last timestep)
    """
    # Average Displacement Error
    displacement_error = torch.norm(predictions - ground_truth, dim=2)  # Euclidean distance at each timestep
    ade = displacement_error.mean().item()
    
    # Final Displacement Error
    fde = displacement_error[:, -1].mean().item()
    
    return {
        'ADE': ade,
        'FDE': fde
    }