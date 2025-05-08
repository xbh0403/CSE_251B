import torch
import torch.nn as nn
from tqdm import tqdm

def train_model(model, train_loader, val_loader, num_epochs=30, fold_idx=None):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    # To check learning rate: scheduler.get_last_lr()
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f'Fold {fold_idx} Epoch {epoch+1}/{num_epochs} Training')
        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            history = batch['history'].to(device)
            future = batch['future'].to(device)
            # Forward pass
            optimizer.zero_grad()
            
            # Prepare input data dictionary
            data = {'history': history}
            
            # Get predictions
            predictions = model(data)
            # Calculate loss
            loss = criterion(predictions, future)
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
        # Calculate average training loss
        avg_train_loss = train_loss / len(train_loader)
        print(f'Fold {fold_idx} Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}')
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        val_progress_bar = tqdm(val_loader, desc=f'Fold {fold_idx} Epoch {epoch+1}/{num_epochs} Validation')
        with torch.no_grad():
            for batch in val_progress_bar:
                # Move data to device
                history = batch['history'].to(device)
                future = batch['future'].to(device)
                
                # Prepare input data dictionary
                data = {'history': history}
                
                # Get predictions
                predictions = model(data)
                
                # Calculate loss
                loss = criterion(predictions, future)
                val_loss += loss.item()
                val_progress_bar.set_postfix({'val_loss': loss.item()})
        
        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_loader)
        print(f'Fold {fold_idx} Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}')
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Save the best model for the current fold
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_save_path = f'best_model_fold_{fold_idx}.pth' if fold_idx is not None else 'best_model.pth'
            torch.save(model.state_dict(), model_save_path)
            print(f'Model for fold {fold_idx} saved to {model_save_path} with validation loss: {avg_val_loss:.4f}')
    
    return model, best_val_loss