import torch
import numpy as np
import matplotlib.pyplot as plt

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
    
    # Timestep-wise ADE
    timestep_ade = displacement_error.mean(dim=0)
    
    return {
        'ADE': ade,
        'FDE': fde,
        'timestep_ade': timestep_ade.detach().cpu().numpy()
    }

def compare_with_baseline(model, baseline_fn, val_loader, val_dataset):
    """
    Compare model predictions with constant velocity baseline
    
    Args:
        model: Trained model
        baseline_fn: Function to compute baseline predictions
        val_loader: DataLoader for validation data
        val_dataset: Original validation dataset for denormalization
        
    Returns:
        Dictionary of comparison metrics
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    model_metrics = {'ADE': [], 'FDE': [], 'timestep_ade': []}
    baseline_metrics = {'ADE': [], 'FDE': [], 'timestep_ade': []}
    
    with torch.no_grad():
        for batch in val_loader:
            # Handle different batch structures
            if isinstance(batch, dict):
                history = batch['history'].to(device)
                future = batch['future'].to(device)
            else:
                history, future = batch
                history = history.to(device)
                future = future.to(device)
            
            # Prepare input data dictionary
            data = {'history': history}
            
            # Get model predictions
            model_pred = model(data)
            
            # Get baseline predictions
            baseline_pred = baseline_fn(history)
            
            # Compute metrics for model
            model_batch_metrics = compute_metrics(model_pred, future, val_dataset)
            
            # Compute metrics for baseline
            baseline_batch_metrics = compute_metrics(baseline_pred, future, val_dataset)
            
            # Store metrics
            for key in ['ADE', 'FDE']:
                model_metrics[key].append(model_batch_metrics[key])
                baseline_metrics[key].append(baseline_batch_metrics[key])
            
            if len(model_metrics['timestep_ade']) == 0:
                model_metrics['timestep_ade'] = model_batch_metrics['timestep_ade']
                baseline_metrics['timestep_ade'] = baseline_batch_metrics['timestep_ade']
            else:
                model_metrics['timestep_ade'] = (model_metrics['timestep_ade'] + 
                                               model_batch_metrics['timestep_ade']) / 2
                baseline_metrics['timestep_ade'] = (baseline_metrics['timestep_ade'] + 
                                                  baseline_batch_metrics['timestep_ade']) / 2
    
    # Compute average metrics
    for key in ['ADE', 'FDE']:
        model_metrics[key] = np.mean(model_metrics[key])
        baseline_metrics[key] = np.mean(baseline_metrics[key])
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.plot(model_metrics['timestep_ade'], label='Model')
    plt.plot(baseline_metrics['timestep_ade'], label='Constant Velocity')
    plt.xlabel('Future Timestep')
    plt.ylabel('ADE')
    plt.title('Model vs. Constant Velocity Baseline')
    plt.legend()
    plt.savefig('model_vs_baseline.png')
    
    return {
        'model': model_metrics,
        'baseline': baseline_metrics
    }