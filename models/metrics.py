import torch

def compute_metrics(predictions, ground_truth, test_dataset):
    """
    Compute trajectory prediction metrics using denormalized values
    
    Args:
        predictions: Normalized predicted trajectories [batch_size, seq_len, 2]
        ground_truth: Normalized ground truth trajectories [batch_size, seq_len, 2]
        test_dataset: Dataset object with denormalization method
        
    Returns:
        Dictionary of metrics
    """
    # Denormalize both predictions and ground truth
    predictions_denorm = test_dataset.denormalize_positions(predictions)
    ground_truth_denorm = test_dataset.denormalize_positions(ground_truth)
    
    # Average Displacement Error (using denormalized values)
    displacement_error = torch.norm(predictions_denorm - ground_truth_denorm, dim=2)
    ade = displacement_error.mean().item()
    
    # Final Displacement Error (using denormalized values)
    fde = displacement_error[:, -1].mean().item()
    
    return {
        'ADE': ade,
        'FDE': fde
    }