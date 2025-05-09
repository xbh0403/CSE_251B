import os
import torch
import torch.nn as nn
from tqdm import tqdm
from data_utils.feature_engineering import compute_constant_velocity

# Add this to your training code before the main training loop
def pretrain_for_residuals(model, train_loader, num_epochs=5):
    """Pre-train model to predict zero residuals"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        
        for batch in train_loader:
            if isinstance(batch, dict):
                history = batch['history'].to(device)
                future = batch['future'].to(device)
            else:
                history, future = batch
                history = history.to(device)
                future = future.to(device)
            
            # Get constant velocity predictions
            const_vel_pred = compute_constant_velocity(history)
            
            # Target residuals should be zero
            target_residuals = torch.zeros_like(const_vel_pred)
            
            # Forward pass
            optimizer.zero_grad()
            data = {'history': history}
            predicted_residuals = model(data) - const_vel_pred  # Ensure we're training to predict zero residuals
            
            # Loss based on residual prediction
            loss = criterion(predicted_residuals, target_residuals)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f'Pretraining Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}')
    
    return model


def train_model(model, train_loader, val_loader, train_original_dataset, val_original_dataset, num_epochs=30):
    """
    Train the model and track both normalized and denormalized MSE losses
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        train_original_dataset: Original TrajectoryDataset for denormalization (not Subset)
        val_original_dataset: Original TrajectoryDataset for denormalization (not Subset)
        num_epochs: Number of epochs to train
    """
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.2, patience=3
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    # For logging
    train_losses = {'normalized': [], 'denormalized': []}
    val_losses = {'normalized': [], 'denormalized': []}
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss_norm = 0.0
        train_loss_denorm = 0.0
        
        for batch_idx, batch in enumerate(tqdm(train_loader)):
            # For Subset objects, the __getitem__ returns whatever the dataset returns
            # Handle accordingly based on your TrajectoryDataset implementation
            
            # With Subset objects from random_split, the batch might look different
            if isinstance(batch, dict):
                history = batch['history'].to(device)
                future = batch['future'].to(device)
            else:
                # If your batch is a tuple or different structure, handle it accordingly
                # Example if your Subset returns (history, future):
                history, future = batch
                history = history.to(device)
                future = future.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            
            # Prepare input data dictionary
            data = {'history': history}
            
            # Calculate baseline predictions
            baseline_pred = compute_constant_velocity(history)
            
            # Get model predictions (as residuals)
            residuals = model(data)
            
            # Apply residuals to baseline
            predictions = baseline_pred + residuals
            
            # Calculate loss against ground truth
            loss_normalized = criterion(predictions, future)
            
            # Calculate denormalized loss (for information)
            predictions_denorm = train_original_dataset.denormalize_positions(predictions)
            future_denorm = train_original_dataset.denormalize_positions(future)
            loss_denormalized = criterion(predictions_denorm, future_denorm)
            
            # Backward and optimize (using normalized loss)
            loss_normalized.backward()
            optimizer.step()
            
            # Track both losses
            train_loss_norm += loss_normalized.item()
            train_loss_denorm += loss_denormalized.item()
            
        
        # Calculate average training losses
        avg_train_loss_norm = train_loss_norm / len(train_loader)
        avg_train_loss_denorm = train_loss_denorm / len(train_loader)
        
        # Store for plotting
        train_losses['normalized'].append(avg_train_loss_norm)
        train_losses['denormalized'].append(avg_train_loss_denorm)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss (Normalized): {avg_train_loss_norm:.4f}, '
              f'Training Loss (Denormalized): {avg_train_loss_denorm:.2f}')
        
        # Validation phase
        model.eval()
        val_loss_norm = 0.0
        val_loss_denorm = 0.0
        
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
                
                # Calculate baseline predictions
                baseline_pred = compute_constant_velocity(history)
                
                # Get model predictions (as residuals)
                residuals = model(data)
                
                # Apply residuals to baseline
                predictions = baseline_pred + residuals
                
                # Calculate normalized loss
                loss_normalized = criterion(predictions, future)
                val_loss_norm += loss_normalized.item()
                
                # Calculate denormalized loss
                predictions_denorm = val_original_dataset.denormalize_positions(predictions)
                future_denorm = val_original_dataset.denormalize_positions(future)
                loss_denormalized = criterion(predictions_denorm, future_denorm)
                val_loss_denorm += loss_denormalized.item()
        
        # Calculate average validation losses
        avg_val_loss_norm = val_loss_norm / len(val_loader)
        avg_val_loss_denorm = val_loss_denorm / len(val_loader)
        
        # Store for plotting
        val_losses['normalized'].append(avg_val_loss_norm)
        val_losses['denormalized'].append(avg_val_loss_denorm)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss (Normalized): {avg_val_loss_norm:.4f}, '
              f'Validation Loss (Denormalized): {avg_val_loss_denorm:.2f}')
        
        # Learning rate scheduling (based on normalized validation loss)
        scheduler.step(avg_val_loss_norm)
        
        # Save the best model (based on normalized validation loss)
        os.makedirs('/tscc/nfs/home/bax001/scratch/CSE_251B/checkpoints', exist_ok=True)
        if avg_val_loss_norm < best_val_loss:
            best_val_loss = avg_val_loss_norm
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'norm_val_loss': avg_val_loss_norm,
                'denorm_val_loss': avg_val_loss_denorm
            }, '/tscc/nfs/home/bax001/scratch/CSE_251B/checkpoints/best_model.pth')
            print(f'Model saved with validation loss: {avg_val_loss_norm:.4f} (normalized), '
                  f'{avg_val_loss_denorm:.2f} (denormalized)')
    
    return model


def debug_predictions(model, val_loader, val_dataset, num_examples=5):
    """
    Generate and visualize predictions vs ground truth for debugging
    
    Args:
        model: Trained model
        val_loader: DataLoader for validation data
        val_dataset: Original validation dataset for denormalization
        num_examples: Number of examples to visualize
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from data_utils.feature_engineering import compute_constant_velocity
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Create output directory for visualizations
    os.makedirs('debug_visualizations', exist_ok=True)
    
    # Process a few batches for visualization
    example_count = 0
    
    # Calculate error metrics
    model_ade_sum = 0.0
    baseline_ade_sum = 0.0
    model_fde_sum = 0.0
    baseline_fde_sum = 0.0
    
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
            
            # Get baseline predictions
            baseline_pred = compute_constant_velocity(history)
            
            # Get model predictions
            residuals = model(data)
            model_pred = baseline_pred + residuals
            
            # Denormalize for visualization
            history_denorm = val_dataset.denormalize_positions(history[:, 0, :, :2])
            future_denorm = val_dataset.denormalize_positions(future)
            model_pred_denorm = val_dataset.denormalize_positions(model_pred)
            baseline_pred_denorm = val_dataset.denormalize_positions(baseline_pred)
            
            # Calculate ADE and FDE for this batch
            for i in range(history.shape[0]):
                # Calculate ADE (Average Displacement Error)
                model_ade = torch.mean(torch.sqrt(torch.sum((model_pred_denorm[i] - future_denorm[i])**2, dim=1)))
                baseline_ade = torch.mean(torch.sqrt(torch.sum((baseline_pred_denorm[i] - future_denorm[i])**2, dim=1)))
                
                # Calculate FDE (Final Displacement Error)
                model_fde = torch.sqrt(torch.sum((model_pred_denorm[i, -1] - future_denorm[i, -1])**2))
                baseline_fde = torch.sqrt(torch.sum((baseline_pred_denorm[i, -1] - future_denorm[i, -1])**2))
                
                model_ade_sum += model_ade.item()
                baseline_ade_sum += baseline_ade.item()
                model_fde_sum += model_fde.item()
                baseline_fde_sum += baseline_fde.item()
            
            # Visualize a few examples
            for i in range(min(history.shape[0], num_examples - example_count)):
                plt.figure(figsize=(12, 8))
                
                # Extract the first agent's history (assuming history shape is [batch, agents, time, features])
                # If history shape is different, adjust accordingly
                if history_denorm.dim() > 3:  # If history includes multiple agents
                    agent_history = history_denorm[i, 0, :, :2].cpu().numpy()
                else:  # If history is just for one agent
                    agent_history = history_denorm[i, :, :2].cpu().numpy()
                
                # Plot history
                plt.plot(agent_history[:, 0], agent_history[:, 1], 'ko-', label='History', markersize=4)
                
                # Plot ground truth future
                future_path = future_denorm[i, :, :2].cpu().numpy()
                plt.plot(future_path[:, 0], future_path[:, 1], 'go-', label='Ground Truth', markersize=4)
                
                # Plot model prediction
                model_path = model_pred_denorm[i, :, :2].cpu().numpy()
                plt.plot(model_path[:, 0], model_path[:, 1], 'bo-', label='Model Prediction', markersize=4)
                
                # Plot baseline prediction
                baseline_path = baseline_pred_denorm[i, :, :2].cpu().numpy()
                plt.plot(baseline_path[:, 0], baseline_path[:, 1], 'ro-', label='Constant Velocity', markersize=4)
                
                # Calculate errors for this example
                model_ade_i = np.mean(np.sqrt(np.sum((model_path - future_path)**2, axis=1)))
                baseline_ade_i = np.mean(np.sqrt(np.sum((baseline_path - future_path)**2, axis=1)))
                model_fde_i = np.sqrt(np.sum((model_path[-1] - future_path[-1])**2))
                baseline_fde_i = np.sqrt(np.sum((baseline_path[-1] - future_path[-1])**2))
                
                plt.xlabel('X Position')
                plt.ylabel('Y Position')
                plt.title(f'Trajectory Example {example_count + 1}\n'
                          f'Model ADE: {model_ade_i:.2f}, FDE: {model_fde_i:.2f}\n'
                          f'Baseline ADE: {baseline_ade_i:.2f}, FDE: {baseline_fde_i:.2f}')
                plt.legend()
                plt.grid(True)
                
                # Add markers for start and end points
                plt.scatter(agent_history[0, 0], agent_history[0, 1], c='k', s=100, marker='*', label='Start')
                plt.scatter(future_path[-1, 0], future_path[-1, 1], c='g', s=100, marker='*', label='GT End')
                plt.scatter(model_path[-1, 0], model_path[-1, 1], c='b', s=100, marker='*', label='Model End')
                plt.scatter(baseline_path[-1, 0], baseline_path[-1, 1], c='r', s=100, marker='*', label='Baseline End')
                
                # Save with higher resolution
                plt.savefig(f'debug_visualizations/trajectory_example_{example_count + 1}.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                example_count += 1
            
            if example_count >= num_examples:
                break
    
    # Print overall metrics
    total_examples = min(example_count, num_examples)
    if total_examples > 0:
        print(f"\nOverall metrics for {total_examples} examples:")
        print(f"Model ADE: {model_ade_sum/total_examples:.2f}, FDE: {model_fde_sum/total_examples:.2f}")
        print(f"Baseline ADE: {baseline_ade_sum/total_examples:.2f}, FDE: {baseline_fde_sum/total_examples:.2f}")
        print(f"Improvement: {((baseline_ade_sum - model_ade_sum)/baseline_ade_sum)*100:.2f}% in ADE, "
              f"{((baseline_fde_sum - model_fde_sum)/baseline_fde_sum)*100:.2f}% in FDE")