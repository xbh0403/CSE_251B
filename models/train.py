import os
import torch
import torch.nn as nn
from tqdm import tqdm

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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
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
            
            # Get predictions (normalized)
            predictions = model(data)
            
            # Calculate normalized loss (for backpropagation)
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
                
                # Get predictions
                predictions = model(data)
                
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