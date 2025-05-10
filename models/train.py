import torch
import torch.nn as nn
from tqdm import tqdm

def train_model(model, train_loader, val_loader, num_epochs=100, early_stopping_patience=10, 
                lr=1e-3, weight_decay=1e-4, lr_step_size=20, lr_gamma=0.25):
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
        
        for batch in train_loader:
            # Move data to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            predictions = model(batch)
            
            # Calculate loss
            loss = criterion(predictions, batch['future'])
            
            # Backward and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # Calculate average training loss
        train_loss /= len(train_loader)
        
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
                
                # Forward pass
                predictions = model(batch)
                
                # Calculate normalized loss
                val_loss += criterion(predictions, batch['future']).item()
                
                # Calculate unnormalized metrics
                pred_unnorm = predictions * batch['scale'].view(-1, 1, 1)
                future_unnorm = batch['future'] * batch['scale'].view(-1, 1, 1)
                
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
            'val_mse': f"{val_loss:.4f}",
            'val_mae': f"{val_mae:.4f}",
            'val_mse': f"{val_mse:.4f}"
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