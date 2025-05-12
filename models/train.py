import torch
import torch.nn as nn
from tqdm import tqdm

def train_model(model, train_loader, val_loader, num_epochs=100, early_stopping_patience=10,
                lr=1e-3, weight_decay=1e-4, lr_step_size=20, lr_gamma=0.25):
    """
    Train the model with validation, handling padded history via mask.
    """
    # Device configuration
    device = model.device if hasattr(model, 'device') else next(model.parameters()).device

    # Loss, optimizer, and scheduler
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)

    best_val_loss = float('inf')
    no_improvement = 0

    progress_bar = tqdm(range(num_epochs), desc="Epoch", unit="epoch")
    for epoch in progress_bar:
        # Training phase
        model.train()
        train_loss = 0.0
        train_mse_unnorm = 0.0

        for batch in train_loader:
            # Move tensors to device
            history = batch['history'].to(device)   # (B, N_agents, T_hist, features)
            future = batch['future'].to(device)     # (B, T_fut, 2)
            mask = batch['mask'].to(device)         # (B, N_agents, T_hist)
            scale = batch['scale'].to(device)

            optimizer.zero_grad()
            # Forward: pass mask so model can ignore padded history
            predictions = model(history, mask)
            loss = criterion(predictions, future)

            # Track unnormalized MSE
            pred_unnorm = predictions * scale.view(-1, 1, 1)
            future_unnorm = future * scale.view(-1, 1, 1)
            train_mse_unnorm += nn.MSELoss()(pred_unnorm, future_unnorm).item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_mse_unnorm /= len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        val_mse = 0.0

        with torch.no_grad():
            for batch in val_loader:
                history = batch['history'].to(device)
                future = batch['future'].to(device)
                mask = batch['mask'].to(device)
                scale = batch['scale'].to(device)

                predictions = model(history, mask)
                val_loss += criterion(predictions, future).item()

                pred_unnorm = predictions * scale.view(-1, 1, 1)
                future_unnorm = future * scale.view(-1, 1, 1)
                val_mae += nn.L1Loss()(pred_unnorm, future_unnorm).item()
                val_mse += nn.MSELoss()(pred_unnorm, future_unnorm).item()

        val_loss /= len(val_loader)
        val_mae /= len(val_loader)
        val_mse /= len(val_loader)

        scheduler.step()

        progress_bar.set_postfix({
            'lr': f"{optimizer.param_groups[0]['lr']:.6f}",
            'train_loss': f"{train_loss:.4f}",
            'train_mse_unnorm': f"{train_mse_unnorm:.4f}",
            'val_loss': f"{val_loss:.4f}",
            'val_mae': f"{val_mae:.4f}",
            'val_mse': f"{val_mse:.4f}"
        })

        # Early stopping / checkpointing
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

    # Load best checkpoint
    checkpoint = torch.load("best_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])

    return model, checkpoint
