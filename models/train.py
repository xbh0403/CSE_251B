import torch
import torch.nn as nn
from tqdm import tqdm

def train_model(model, train_loader, val_loader, num_epochs=100, early_stopping_patience=10, 
                lr=1e-3, weight_decay=1e-4, lr_step_size=20, lr_gamma=0.25, teacher_forcing_ratio=0.5):
    """
    Train the model with validation
    
    Args:
        model: The model to train
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
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)
    
    # Training loop
    best_val_loss = float('inf')
    no_improvement = 0
    
    progress_bar = tqdm(range(num_epochs), desc="Epoch", unit="epoch")
    for epoch in progress_bar:
        # Training phase
        model.train()
        train_loss = 0.0
        train_mse_unnorm = 0.0
        
        for batch in train_loader:
            # Move data to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            # Forward pass (with teacher forcing during training)
            optimizer.zero_grad()
            predictions = model(batch, teacher_forcing_ratio=teacher_forcing_ratio)
            
            # Calculate loss
            loss = criterion(predictions, batch['future'])
            
            # Calculate unnormalized training MSE
            pred_unnorm = predictions * batch['scale_position'].view(-1, 1, 1)
            future_unnorm = batch['future'] * batch['scale_position'].view(-1, 1, 1)
            train_mse_unnorm += nn.MSELoss()(pred_unnorm, future_unnorm).item()
            
            # Backward and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # Calculate average training loss
        train_loss /= len(train_loader)
        train_mse_unnorm /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        val_mse = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move data to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(device)
                
                # Forward pass (no teacher forcing during validation)
                predictions = model(batch, teacher_forcing_ratio=0.0)
                
                # Calculate normalized loss
                val_loss += criterion(predictions, batch['future']).item()
                
                # Calculate unnormalized metrics
                pred_unnorm = predictions * batch['scale_position'].view(-1, 1, 1)
                future_unnorm = batch['future'] * batch['scale_position'].view(-1, 1, 1)
                
                val_mae += nn.L1Loss()(pred_unnorm, future_unnorm).item()
                val_mse += nn.MSELoss()(pred_unnorm, future_unnorm).item()
        
        # Calculate average validation losses
        val_loss /= len(val_loader)
        val_mae /= len(val_loader)
        val_mse /= len(val_loader)
        
        # Update learning rate
        scheduler.step()
        
        # Update progress bar with metrics
        progress_bar.set_postfix({
            'lr': f"{optimizer.param_groups[0]['lr']:.6f}",
            'train_mse': f"{train_loss:.4f}",
            'train_mse_unnorm': f"{train_mse_unnorm:.4f}",
            'val_mse': f"{val_loss:.4f}",
            'val_mae': f"{val_mae:.4f}",
            'val_mse_unnorm': f"{val_mse:.4f}"
        })
        
        # Save the best model
        if val_loss < best_val_loss - 1e-3:
            best_val_loss = val_loss
            no_improvement = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_mae': val_mae,
                'val_mse': val_mse
            }, "best_model.pth")
        else:
            no_improvement += 1
            if no_improvement >= early_stopping_patience:
                progress_bar.write("Early stopping!")
                break
    
    # Load the best model
    checkpoint = torch.load("best_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, checkpoint


def train_multimodal_model(model, train_loader, val_loader, num_epochs=100, early_stopping_patience=10, 
                lr=1e-3, weight_decay=1e-4, lr_step_size=20, lr_gamma=0.25, teacher_forcing_ratio=0.5):
    """
    Train a multi-modal trajectory prediction model
    """
    # Device configuration
    device = model.device if hasattr(model, 'device') else next(model.parameters()).device
    
    # Optimization
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)
    
    # Training loop
    best_val_loss = float('inf')
    no_improvement = 0
    
    progress_bar = tqdm(range(num_epochs), desc="Epoch", unit="epoch")
    for epoch in progress_bar:
        # Training phase
        model.train()
        train_loss = 0.0
        train_mse_unnorm = 0.0
        
        for batch in train_loader:
            # Move data to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            # Forward pass (with teacher forcing during training)
            optimizer.zero_grad()
            predictions, confidences = model(batch, teacher_forcing_ratio=teacher_forcing_ratio)
            
            # Expand ground truth to compare with multi-modal predictions
            gt = batch['future'].unsqueeze(2).expand(-1, -1, model.num_modes, -1)
            
            # Calculate loss using winner-takes-all approach
            # Calculate MSE for each mode
            per_mode_mse = torch.mean((predictions - gt)**2, dim=(1, 3))  # [batch_size, num_modes]
            
            # Weight the MSE by confidence scores
            weighted_mse = per_mode_mse * confidences
            
            # Sum across modes
            loss = torch.mean(torch.sum(weighted_mse, dim=1))
            
            # Calculate minimum MSE across all modes (best mode)
            min_mode_mse = torch.min(per_mode_mse, dim=1)[0]  # Get the best mode's error
            best_mode_loss = torch.mean(min_mode_mse)
            
            # Combined loss: weighted MSE + best mode MSE
            combined_loss = loss + best_mode_loss
            
            # Calculate unnormalized best mode MSE for metrics
            # Get best mode index for each batch item
            best_modes = torch.argmin(per_mode_mse, dim=1)
            
            # Select best predictions
            best_predictions = torch.gather(
                predictions, 
                dim=2, 
                index=best_modes.view(-1, 1, 1, 1).expand(-1, predictions.size(1), 1, predictions.size(-1))
            ).squeeze(2)  # [batch_size, seq_len, output_dim]
            
            # Calculate unnormalized MSE
            pred_unnorm = best_predictions * batch['scale_position'].view(-1, 1, 1)
            future_unnorm = batch['future'] * batch['scale_position'].view(-1, 1, 1)
            train_mse_unnorm += nn.MSELoss()(pred_unnorm, future_unnorm).item()
            
            # Backward and optimize
            combined_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            train_loss += combined_loss.item()
        
        # Calculate average training loss
        train_loss /= len(train_loader)
        train_mse_unnorm /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        val_mse = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move data to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(device)
                
                # Forward pass (no teacher forcing during validation)
                predictions, confidences = model(batch, teacher_forcing_ratio=0.0)
                
                # Calculate per-mode MSE
                gt = batch['future'].unsqueeze(2).expand(-1, -1, model.num_modes, -1)
                per_mode_mse = torch.mean((predictions - gt)**2, dim=(1, 3))
                
                # Get best mode for each sample
                best_modes = torch.argmin(per_mode_mse, dim=1)
                
                # Select best predictions
                best_predictions = torch.gather(
                    predictions, 
                    dim=2, 
                    index=best_modes.view(-1, 1, 1, 1).expand(-1, predictions.size(1), 1, predictions.size(-1))
                ).squeeze(2)
                
                # Calculate normalized best mode loss
                val_loss += nn.MSELoss()(best_predictions, batch['future']).item()
                
                # Calculate unnormalized metrics
                pred_unnorm = best_predictions * batch['scale_position'].view(-1, 1, 1)
                future_unnorm = batch['future'] * batch['scale_position'].view(-1, 1, 1)
                
                val_mae += nn.L1Loss()(pred_unnorm, future_unnorm).item()
                val_mse += nn.MSELoss()(pred_unnorm, future_unnorm).item()
        
        # Calculate average validation losses
        val_loss /= len(val_loader)
        val_mae /= len(val_loader)
        val_mse /= len(val_loader)
        
        # Update learning rate
        scheduler.step()
        
        # Update progress bar with metrics
        progress_bar.set_postfix({
            'lr': f"{optimizer.param_groups[0]['lr']:.6f}",
            'train_loss': f"{train_loss:.4f}",
            'train_mse_unnorm': f"{train_mse_unnorm:.4f}",
            'val_mse': f"{val_loss:.4f}",
            'val_mae': f"{val_mae:.4f}",
            'val_mse_unnorm': f"{val_mse:.4f}"
        })
        
        # Save the best model
        if val_loss < best_val_loss - 1e-3:
            best_val_loss = val_loss
            no_improvement = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_mae': val_mae,
                'val_mse': val_mse
            }, "best_model.pth")
        else:
            no_improvement += 1
            if no_improvement >= early_stopping_patience:
                progress_bar.write("Early stopping!")
                break
    
    # Load the best model
    checkpoint = torch.load("best_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, checkpoint