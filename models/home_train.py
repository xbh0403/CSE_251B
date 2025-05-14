# models/home_train.py

import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

def train_home_model(model, train_loader, val_loader, num_epochs=100, early_stopping_patience=10, 
                     lr=1e-3, weight_decay=1e-4, lr_step_size=20, lr_gamma=0.25):
    """
    Train the HOME model
    
    Args:
        model: HOME model
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
    device = next(model.parameters()).device
    
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
        train_loss = 0.0
        
        for batch in train_loader:
            # Move data to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            
            # Get object types - most common non-zero type
            obj_types = batch['history'][:, 0, 0, 5].long()  # Use first timestep's type
            obj_types = torch.clamp(obj_types, min=0, max=9)  # Ensure valid types
            
            # Teacher forcing during training
            predictions, confidences = model(batch, teacher_forcing_ratio=0.5)
            
            # Calculate loss
            loss = model.compute_loss(predictions, confidences, batch['future'], obj_types)
            
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
        val_ade = 0.0
        val_fde = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move data to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(device)
                
                # Get object types
                obj_types = batch['history'][:, 0, 0, 5].long()
                obj_types = torch.clamp(obj_types, min=0, max=9)
                
                # Forward pass (no teacher forcing during validation)
                predictions, confidences = model(batch, teacher_forcing_ratio=0.0)
                
                # Calculate loss
                loss = model.compute_loss(predictions, confidences, batch['future'], obj_types)
                val_loss += loss.item()
                
                # Calculate metrics (minimum across all samples)
                # ADE (Average Displacement Error)
                displacement_all = torch.norm(
                    predictions - batch['future'].unsqueeze(1), dim=3
                )  # [batch_size, num_samples, seq_len]
                
                ade_all = torch.mean(displacement_all, dim=2)  # [batch_size, num_samples]
                min_ade, _ = torch.min(ade_all, dim=1)  # [batch_size]
                val_ade += torch.mean(min_ade).item()
                
                # FDE (Final Displacement Error)
                final_displacement_all = torch.norm(
                    predictions[:, :, -1] - batch['future'][:, -1].unsqueeze(1), dim=2
                )  # [batch_size, num_samples]
                
                min_fde, _ = torch.min(final_displacement_all, dim=1)  # [batch_size]
                val_fde += torch.mean(min_fde).item()
        
        # Calculate average validation metrics
        val_loss /= len(val_loader)
        val_ade /= len(val_loader)
        val_fde /= len(val_loader)
        
        # Update learning rate
        scheduler.step()
        
        # Update progress bar with metrics
        progress_bar.set_postfix({
            'lr': f"{optimizer.param_groups[0]['lr']:.6f}",
            'train_loss': f"{train_loss:.4f}",
            'val_loss': f"{val_loss:.4f}",
            'val_ade': f"{val_ade:.4f}",
            'val_fde': f"{val_fde:.4f}"
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
                'val_ade': val_ade,
                'val_fde': val_fde
            }, "home_best_model.pth")
        else:
            no_improvement += 1
            if no_improvement >= early_stopping_patience:
                progress_bar.write("Early stopping!")
                break
    
    # Load the best model
    checkpoint = torch.load("home_best_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, checkpoint