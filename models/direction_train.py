import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

def train_direction_aware_model(model, train_loader, val_loader, num_epochs=100, early_stopping_patience=10, 
                lr=1e-3, weight_decay=1e-4, lr_step_size=20, lr_gamma=0.25, teacher_forcing_ratio=0.5):
    """
    Train the direction-aware model with multi-component loss
    
    Args:
        model: The direction-aware model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Maximum number of epochs to train
        early_stopping_patience: Number of epochs to wait for improvement
        lr: Initial learning rate
        weight_decay: Weight decay for optimizer
        lr_step_size: Epochs between learning rate reductions
        lr_gamma: Factor to reduce learning rate
        teacher_forcing_ratio: Probability of using teacher forcing during training (0-1)
    """
    # Device configuration
    device = model.device if hasattr(model, 'device') else next(model.parameters()).device
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)
    
    # Training loop
    best_val_loss = float('inf')
    no_improvement = 0
    
    progress_bar = tqdm(range(num_epochs), desc="Epoch", unit="epoch")
    for epoch in progress_bar:
        # Training phase
        model.train()
        train_total_loss = 0.0
        train_mse_loss = 0.0
        train_direction_loss = 0.0
        train_initial_direction_loss = 0.0
        train_mse_unnorm = 0.0
        
        for batch in train_loader:
            # Move data to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            # Forward pass (with teacher forcing during training)
            optimizer.zero_grad()
            predictions = model(batch, teacher_forcing_ratio=teacher_forcing_ratio)
            
            # Calculate multi-component loss
            losses = model.compute_losses(
                predictions, 
                batch['future'],
                batch['history'][:, 0, :, :]  # Focal agent history
            )
            
            # Calculate unnormalized MSE for tracking
            pred_unnorm = predictions * batch['scale'].view(-1, 1, 1)
            future_unnorm = batch['future'] * batch['scale'].view(-1, 1, 1)
            train_mse_unnorm += nn.MSELoss()(pred_unnorm, future_unnorm).item()
            
            # Backward and optimize
            loss = losses['total_loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            # Accumulate losses
            train_total_loss += losses['total_loss'].item()
            train_mse_loss += losses['mse_loss'].item()
            train_direction_loss += losses['direction_loss'].item()
            train_initial_direction_loss += losses['initial_direction_loss'].item()
        
        # Calculate average training loss
        num_batches = len(train_loader)
        train_total_loss /= num_batches
        train_mse_loss /= num_batches
        train_direction_loss /= num_batches
        train_initial_direction_loss /= num_batches
        train_mse_unnorm /= num_batches
        
        # Validation phase
        model.eval()
        val_total_loss = 0.0
        val_mse_loss = 0.0
        val_direction_loss = 0.0
        val_initial_direction_loss = 0.0
        val_mae = 0.0
        val_mse_unnorm = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move data to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(device)
                
                # Forward pass (no teacher forcing during validation)
                predictions = model(batch, teacher_forcing_ratio=0.0)
                
                # Calculate multi-component loss
                losses = model.compute_losses(
                    predictions, 
                    batch['future'],
                    batch['history'][:, 0, :, :]  # Focal agent history
                )
                
                # Calculate unnormalized metrics
                pred_unnorm = predictions * batch['scale'].view(-1, 1, 1)
                future_unnorm = batch['future'] * batch['scale'].view(-1, 1, 1)
                
                val_mae += nn.L1Loss()(pred_unnorm, future_unnorm).item()
                val_mse_unnorm += nn.MSELoss()(pred_unnorm, future_unnorm).item()
                
                # Accumulate losses
                val_total_loss += losses['total_loss'].item()
                val_mse_loss += losses['mse_loss'].item()
                val_direction_loss += losses['direction_loss'].item()
                val_initial_direction_loss += losses['initial_direction_loss'].item()
        
        # Calculate average validation losses
        num_val_batches = len(val_loader)
        val_total_loss /= num_val_batches
        val_mse_loss /= num_val_batches
        val_direction_loss /= num_val_batches
        val_initial_direction_loss /= num_val_batches
        val_mae /= num_val_batches
        val_mse_unnorm /= num_val_batches
        
        # Update learning rate
        scheduler.step()
        
        # Update progress bar with metrics
        progress_bar.set_postfix({
            'lr': f"{optimizer.param_groups[0]['lr']:.6f}",
            'train_loss': f"{train_total_loss:.4f}",
            'val_loss': f"{val_total_loss:.4f}",
            'dir_loss': f"{val_direction_loss:.4f}",
            'init_dir': f"{val_initial_direction_loss:.4f}",
            'val_mae': f"{val_mae:.4f}"
        })
        
        # Save the best model
        if val_total_loss < best_val_loss - 1e-3:
            best_val_loss = val_total_loss
            no_improvement = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_total_loss': val_total_loss,
                'val_mse_loss': val_mse_loss,
                'val_direction_loss': val_direction_loss,
                'val_initial_direction_loss': val_initial_direction_loss,
                'val_mae': val_mae,
                'val_mse_unnorm': val_mse_unnorm
            }, "best_direction_model.pth")
            
            # Also save detailed metrics
            metrics = {
                'train_metrics': {
                    'total_loss': train_total_loss,
                    'mse_loss': train_mse_loss,
                    'direction_loss': train_direction_loss,
                    'initial_direction_loss': train_initial_direction_loss,
                    'mse_unnorm': train_mse_unnorm
                },
                'val_metrics': {
                    'total_loss': val_total_loss,
                    'mse_loss': val_mse_loss,
                    'direction_loss': val_direction_loss,
                    'initial_direction_loss': val_initial_direction_loss,
                    'mae': val_mae,
                    'mse_unnorm': val_mse_unnorm
                }
            }
            np.save("best_model_metrics.npy", metrics)
        else:
            no_improvement += 1
            if no_improvement >= early_stopping_patience:
                progress_bar.write(f"Early stopping at epoch {epoch}!")
                break
    
    # Load the best model
    checkpoint = torch.load("best_direction_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, checkpoint