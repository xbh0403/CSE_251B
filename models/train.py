import torch
import torch.nn as nn
from tqdm import tqdm
import json
import os
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# def save_training_metrics(metrics_dict, model_name="model", save_dir="logs"):
#     """
#     Save training metrics to CSV and JSON files and create visualization plots.
    
#     Args:
#         metrics_dict: Dictionary containing training/validation metrics by epoch
#         model_name: Name to use in saved files
#         save_dir: Directory to save metrics files
#     """
#     # Create logs directory if it doesn't exist
#     os.makedirs(save_dir, exist_ok=True)
    
#     # Generate a timestamp for the log files
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     base_filename = f"{model_name}_{timestamp}"
    
#     # Convert metrics dictionary to DataFrame for easier manipulation
#     metrics_df = pd.DataFrame(metrics_dict)
    
#     # Save metrics to CSV
#     csv_path = os.path.join(save_dir, f"{base_filename}_metrics.csv")
#     metrics_df.to_csv(csv_path, index_label='epoch')
#     print(f"Metrics saved to {csv_path}")
    
#     # Save metrics to JSON
#     json_path = os.path.join(save_dir, f"{base_filename}_metrics.json")
#     with open(json_path, 'w') as f:
#         json.dump(metrics_dict, f, indent=2)
#     print(f"Metrics saved to {json_path}")
    
#     # Create visualization directory
#     viz_dir = os.path.join(save_dir, f"{base_filename}_plots")
#     os.makedirs(viz_dir, exist_ok=True)
    
#     # Create training vs validation loss plot
#     plt.figure(figsize=(10, 6))
#     plt.plot(metrics_df['train_loss'], label='Training Loss')
#     plt.plot(metrics_df['val_loss'], label='Validation Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.title(f'{model_name} - Training and Validation Loss')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(os.path.join(viz_dir, 'loss_curves.png'), dpi=300)
#     plt.close()
    
#     # Create MAE/MSE plots for unnormalized metrics
#     plt.figure(figsize=(10, 6))
#     if 'train_mse_unnorm' in metrics_df.columns:
#         plt.plot(metrics_df['train_mse_unnorm'], label='Train MSE')
#     plt.plot(metrics_df['val_mse_unnorm'], label='Val MSE')
#     plt.plot(metrics_df['val_mae'], label='Val MAE')
#     plt.xlabel('Epoch')
#     plt.ylabel('Error (meters)')
#     plt.title(f'{model_name} - Training and Validation Error Metrics')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(os.path.join(viz_dir, 'error_metrics.png'), dpi=300)
#     plt.close()
    
#     # If learning rate is tracked, plot it
#     if 'learning_rate' in metrics_df.columns:
#         plt.figure(figsize=(10, 6))
#         plt.plot(metrics_df['learning_rate'])
#         plt.xlabel('Epoch')
#         plt.ylabel('Learning Rate')
#         plt.title(f'{model_name} - Learning Rate Schedule')
#         plt.yscale('log')
#         plt.grid(True)
#         plt.savefig(os.path.join(viz_dir, 'learning_rate.png'), dpi=300)
#         plt.close()
    
#     # For multimodal models, if physics loss is tracked
#     if 'train_physics_loss' in metrics_df.columns:
#         plt.figure(figsize=(10, 6))
#         plt.plot(metrics_df['train_physics_loss'], label='Train Physics Loss')
#         plt.plot(metrics_df['val_physics_loss'], label='Val Physics Loss')
#         plt.xlabel('Epoch')
#         plt.ylabel('Physics Loss')
#         plt.title(f'{model_name} - Physics Constraint Loss')
#         plt.legend()
#         plt.grid(True)
#         plt.savefig(os.path.join(viz_dir, 'physics_loss.png'), dpi=300)
#         plt.close()
    
#     return csv_path, json_path, viz_dir

def save_training_metrics(metrics_history, model_name, save_dir):
    import os
    import json
    import pandas as pd
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)

    # 1) Convert history dict → DataFrame
    metrics_df = pd.DataFrame(metrics_history)

    # 2) Save raw CSV + JSON
    csv_path = os.path.join(save_dir, f"{model_name}_metrics.csv")
    json_path = os.path.join(save_dir, f"{model_name}_metrics.json")
    metrics_df.to_csv(csv_path, index=False)
    with open(json_path, 'w') as f:
        json.dump(metrics_history, f, indent=2)

    # 3) Plot curves (use the correct keys!)
    viz_dir = os.path.join(save_dir, "plots")
    os.makedirs(viz_dir, exist_ok=True)

    # -- Training Hybrid Loss --
    plt.figure()
    plt.plot(metrics_df['train_hybrid_loss'], label='Training Hybrid Loss')
    plt.plot(metrics_df['val_hybrid_loss'], label='Validation Hybrid Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Hybrid Loss")
    plt.title("Hybrid Loss (MSE + α·MAE) per Epoch")
    plt.legend()
    plt.savefig(os.path.join(viz_dir, "hybrid_loss.png"))
    plt.close()

    # -- Unnormalized MSE, MAE (optional) --
    plt.figure()
    plt.plot(metrics_df['train_mse_unnorm'], label='Train MSE (raw)')
    plt.plot(metrics_df['val_mse_unnorm'], label='Val MSE (raw)')
    plt.xlabel("Epoch")
    plt.ylabel("MSE (physical)")
    plt.title("Unnormalized MSE per Epoch")
    plt.legend()
    plt.savefig(os.path.join(viz_dir, "mse_unnorm.png"))
    plt.close()

    plt.figure()
    plt.plot(metrics_df['val_mae_unnorm'], label='Val MAE (raw)')
    plt.xlabel("Epoch")
    plt.ylabel("MAE (physical)")
    plt.title("Unnormalized MAE (validation) per Epoch")
    plt.legend()
    plt.savefig(os.path.join(viz_dir, "mae_unnorm.png"))
    plt.close()

    return csv_path, json_path, viz_dir

def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs=100,
    early_stopping_patience=10,
    lr=1e-3,
    weight_decay=1e-4,
    lr_step_size=20,
    lr_gamma=0.25,
    teacher_forcing_ratio=0.5,
    model_name="seq2seq_model",
    save_logs=True
):
    """
    使用“混合损失 (MSE + α·MAE)”训练，同时仍然采用 scale_position 反归一化：
    1) 在归一化空间里计算 hybrid_loss = MSE(pred_norm, gt_norm) + α·MAE(pred_norm, gt_norm)
    2) 将 pred_norm * scale_position → pred_raw，计算物理尺度下的 unnorm MSE/MAE
    3) Early‐stopping 依据 hybrid_loss
    """

    device = model.device if hasattr(model, 'device') else next(model.parameters()).device

    # 定义归一化空间下的 MSE / MAE
    mse_criterion = nn.MSELoss(reduction='mean')
    mae_criterion = nn.L1Loss(reduction='mean')
    alpha = 0.5  # 你可以调节 α 的值

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)

    metrics_history = {
        'epoch': [],
        'learning_rate': [],
        'train_hybrid_loss': [],
        'train_mse_unnorm': [],
        'val_hybrid_loss': [],
        'val_mae_unnorm': [],
        'val_mse_unnorm': []
    }

    best_val_loss = float('inf')
    no_improvement = 0

    progress_bar = tqdm(range(num_epochs), desc="Epoch", unit="epoch")
    for epoch in progress_bar:
        # ===== 训练阶段 =====
        model.train()
        running_hybrid_loss = 0.0
        running_mse_unnorm = 0.0

        for batch in train_loader:
            # 把所有 tensor 移到 device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            optimizer.zero_grad()
            # 预测：得到 normalized 空间下 [B,60,2]
            preds_norm = model(batch, teacher_forcing_ratio=teacher_forcing_ratio)

            # —— 1) 归一化空间下的混合损失 —— 
            mse_loss = mse_criterion(preds_norm, batch['future'])
            mae_loss = mae_criterion(preds_norm, batch['future'])
            hybrid_loss = mse_loss + alpha * mae_loss
            running_hybrid_loss += hybrid_loss.item()

            # —— 2) 物理空间下的 unnorm MSE —— 
            #    pred_raw = preds_norm * scale_position
            #    future_raw = batch['future'] * scale_position
            scale_pos = batch['scale_position'].view(-1, 1, 1)  # [B,1,1]
            preds_raw = preds_norm * scale_pos       # [B,60,2]
            gt_raw    = batch['future'] * scale_pos  # [B,60,2]

            # nn.MSELoss(reduction='mean') 本身已经对 60 帧取平均
            mse_phys = mse_criterion(preds_raw, gt_raw)
            running_mse_unnorm += mse_phys.item()

            # —— 3) 反向传播 —— 
            hybrid_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        avg_train_hybrid = running_hybrid_loss / len(train_loader)
        avg_train_mse_unnorm = running_mse_unnorm / len(train_loader)

        # ===== 验证阶段 =====
        model.eval()
        running_val_hybrid = 0.0
        running_val_mae_unnorm = 0.0
        running_val_mse_unnorm = 0.0

        with torch.no_grad():
            for batch in val_loader:
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device)

                preds_norm = model(batch, teacher_forcing_ratio=0.0)

                mse_loss = mse_criterion(preds_norm, batch['future'])
                mae_loss = mae_criterion(preds_norm, batch['future'])
                hybrid_loss = mse_loss + alpha * mae_loss
                running_val_hybrid += hybrid_loss.item()

                scale_pos = batch['scale_position'].view(-1, 1, 1)
                preds_raw = preds_norm * scale_pos       # [B,60,2]
                gt_raw    = batch['future'] * scale_pos  # [B,60,2]

                mae_phys = mae_criterion(preds_raw, gt_raw)
                mse_phys = mse_criterion(preds_raw, gt_raw)
                running_val_mae_unnorm += mae_phys.item()
                running_val_mse_unnorm += mse_phys.item()

        avg_val_hybrid = running_val_hybrid / len(val_loader)
        avg_val_mae_unnorm = running_val_mae_unnorm / len(val_loader)
        avg_val_mse_unnorm = running_val_mse_unnorm / len(val_loader)

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        # 记录指标
        metrics_history['epoch'].append(epoch)
        metrics_history['learning_rate'].append(current_lr)
        metrics_history['train_hybrid_loss'].append(avg_train_hybrid)
        metrics_history['train_mse_unnorm'].append(avg_train_mse_unnorm)
        metrics_history['val_hybrid_loss'].append(avg_val_hybrid)
        metrics_history['val_mae_unnorm'].append(avg_val_mae_unnorm)
        metrics_history['val_mse_unnorm'].append(avg_val_mse_unnorm)

        progress_bar.set_postfix({
            'lr':                 f"{current_lr:.6f}",
            'train_hybrid_loss':  f"{avg_train_hybrid:.4f}",
            'train_mse_unnorm':   f"{avg_train_mse_unnorm:.4f}",
            'val_hybrid_loss':    f"{avg_val_hybrid:.4f}",
            'val_mae_unnorm':     f"{avg_val_mae_unnorm:.4f}"
        })

        # 保存最佳模型
        if avg_val_hybrid < best_val_loss - 1e-3:
            best_val_loss = avg_val_hybrid
            no_improvement = 0
            checkpoint_info = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_hybrid,
                'val_mae': avg_val_mae_unnorm,
                'val_mse': avg_val_mse_unnorm,
                'metrics_history': metrics_history
            }
            torch.save(checkpoint_info, "best_model.pth")
        else:
            no_improvement += 1
            if no_improvement >= early_stopping_patience:
                progress_bar.write("Early stopping!")
                break

    if save_logs:
        csv_path, json_path, viz_dir = save_training_metrics(
            metrics_history,
            model_name=model_name,
            save_dir="logs"
        )
        progress_bar.write(f"Training metrics saved to {csv_path} and {json_path}")
        progress_bar.write(f"Visualization plots saved to {viz_dir}")

    # 加载最佳模型
    checkpoint = torch.load("best_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint, metrics_history




def train_multimodal_model(model, train_loader, val_loader, num_epochs=100, early_stopping_patience=10, 
                lr=1e-3, weight_decay=1e-4, lr_step_size=20, lr_gamma=0.25, teacher_forcing_ratio=0.5,
                physics_weight=0.2, model_name="multimodal_model", save_logs=True):  
    """
    Train a multi-modal trajectory prediction model with physics constraints
    """
    # Device configuration
    device = model.device if hasattr(model, 'device') else next(model.parameters()).device
    
    # Optimization
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)
    
    # Dictionary to store metrics for each epoch
    metrics_history = {
        'epoch': [],
        'learning_rate': [],
        'train_loss': [],
        'train_mse_unnorm': [],
        'train_physics_loss': [],  # Track physics loss separately
        'val_loss': [],
        'val_mae': [],
        'val_mse': [],
        'val_mse_unnorm': [],
        'val_physics_loss': []
    }
    
    # Training loop
    best_val_loss = float('inf')
    no_improvement = 0
    
    progress_bar = tqdm(range(num_epochs), desc="Epoch", unit="epoch")
    for epoch in progress_bar:
        # Training phase
        model.train()
        train_loss = 0.0
        train_mse_unnorm = 0.0
        train_physics_loss = 0.0  # Track physics loss separately
        
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
            pred_loss = torch.mean(torch.sum(weighted_mse, dim=1))
            
            # Calculate minimum MSE across all modes (best mode)
            min_mode_mse = torch.min(per_mode_mse, dim=1)[0]  # Get the best mode's error
            best_mode_loss = torch.mean(min_mode_mse)
            
            # Get best mode predictions for physics loss calculation
            best_modes = torch.argmin(per_mode_mse, dim=1)
            best_predictions = torch.gather(
                predictions, 
                dim=2, 
                index=best_modes.view(-1, 1, 1, 1).expand(-1, predictions.size(1), 1, predictions.size(-1))
            ).squeeze(2)  # [batch_size, seq_len, output_dim]
            
            # Physics constraints loss
            physics_loss = physics_constraint_loss(
                best_predictions, 
                dt=0.1,  # 10 Hz data = 0.1s between frames
                scale_factor=batch['scale_position'][0].item()  # Use the scale factor
            )
            
            # Combined loss: prediction loss + physics loss
            combined_loss = pred_loss + best_mode_loss + physics_weight * physics_loss
            
            # Calculate unnormalized best mode MSE for metrics
            pred_unnorm = best_predictions * batch['scale_position'].view(-1, 1, 1)
            future_unnorm = batch['future'] * batch['scale_position'].view(-1, 1, 1)
            batch_mse_unnorm = nn.MSELoss()(pred_unnorm, future_unnorm).item()
            
            # Backward and optimize
            combined_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            # Track losses
            train_loss += combined_loss.item()
            train_physics_loss += physics_loss.item()
            train_mse_unnorm += batch_mse_unnorm
        
        # Calculate average training loss
        train_loss /= len(train_loader)
        train_mse_unnorm /= len(train_loader)
        train_physics_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        val_mse = 0.0
        val_mse_unnorm = 0.0
        val_physics_loss = 0.0  # Track validation physics loss
        
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
                
                # Calculate physics loss for validation
                physics_loss = physics_constraint_loss(
                    best_predictions, 
                    dt=0.1,
                    scale_factor=batch['scale_position'][0].item()
                )
                val_physics_loss += physics_loss.item()
                
                # Calculate normalized best mode loss
                val_loss += nn.MSELoss()(best_predictions, batch['future']).item()
                
                # Calculate unnormalized metrics
                pred_unnorm = best_predictions * batch['scale_position'].view(-1, 1, 1)
                future_unnorm = batch['future'] * batch['scale_position'].view(-1, 1, 1)
                
                val_mae += nn.L1Loss()(pred_unnorm, future_unnorm).item()
                val_mse_batch = nn.MSELoss()(pred_unnorm, future_unnorm).item()
                val_mse += val_mse_batch
                val_mse_unnorm += val_mse_batch
        
        # Calculate average validation losses
        val_loss /= len(val_loader)
        val_mae /= len(val_loader)
        val_mse /= len(val_loader)
        val_mse_unnorm /= len(val_loader)
        val_physics_loss /= len(val_loader)
        
        # Update learning rate
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        # Store metrics for this epoch
        metrics_history['epoch'].append(epoch)
        metrics_history['learning_rate'].append(current_lr)
        metrics_history['train_loss'].append(train_loss)
        metrics_history['train_mse_unnorm'].append(train_mse_unnorm)
        metrics_history['train_physics_loss'].append(train_physics_loss)
        metrics_history['val_loss'].append(val_loss)
        metrics_history['val_mae'].append(val_mae)
        metrics_history['val_mse'].append(val_mse)
        metrics_history['val_mse_unnorm'].append(val_mse_unnorm)
        metrics_history['val_physics_loss'].append(val_physics_loss)
        
        # Update progress bar with metrics
        progress_bar.set_postfix({
            'lr': f"{current_lr:.6f}",
            'train_loss': f"{train_loss:.4f}",
            'train_phys': f"{train_physics_loss:.4f}",
            'train_mse_unnorm': f"{train_mse_unnorm:.4f}",
            'val_loss': f"{val_loss:.4f}",
            'val_mse': f"{val_mse:.4f}",
            'val_mae': f"{val_mae:.4f}",
            'val_phys': f"{val_physics_loss:.4f}"
        })
        
        # Save the best model
        if val_loss < best_val_loss - 1e-3:
            best_val_loss = val_loss
            no_improvement = 0
            # Include metrics history in checkpoint
            checkpoint_info = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_mae': val_mae,
                'val_mse': val_mse,
                'val_physics_loss': val_physics_loss,
                'metrics_history': metrics_history  # Save metrics history with the checkpoint
            }
            torch.save(checkpoint_info, "best_model.pth")
        else:
            no_improvement += 1
            if no_improvement >= early_stopping_patience:
                progress_bar.write("Early stopping!")
                break
    
    # Save metrics after training
    if save_logs:
        csv_path, json_path, viz_dir = save_training_metrics(
            metrics_history, 
            model_name=model_name,
            save_dir="logs"
        )
        progress_bar.write(f"Training metrics saved to {csv_path} and {json_path}")
        progress_bar.write(f"Visualization plots saved to {viz_dir}")
    
    # Load the best model
    checkpoint = torch.load("best_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, checkpoint, metrics_history

# def train_model(model, train_loader, val_loader, num_epochs=100, early_stopping_patience=10, 
#                 lr=1e-3, weight_decay=1e-4, lr_step_size=20, lr_gamma=0.25, teacher_forcing_ratio=0.5):
#     """
#     Train the model with validation
    
#     Args:
#         model: The model to train
#         train_loader: DataLoader for training data
#         val_loader: DataLoader for validation data
#         num_epochs: Maximum number of epochs to train
#         early_stopping_patience: Number of epochs to wait for improvement
#         lr: Initial learning rate
#         weight_decay: Weight decay for optimizer
#         lr_step_size: Epochs between learning rate reductions
#         lr_gamma: Factor to reduce learning rate
#         teacher_forcing_ratio: Probability of using teacher forcing during training (0-1)
#     """
#     # Device configuration
#     device = model.device if hasattr(model, 'device') else next(model.parameters()).device
    
#     # Loss function and optimizer
#     criterion = nn.MSELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)
    
#     # Training loop
#     best_val_loss = float('inf')
#     no_improvement = 0
    
#     progress_bar = tqdm(range(num_epochs), desc="Epoch", unit="epoch")
#     for epoch in progress_bar:
#         # Training phase
#         model.train()
#         train_loss = 0.0
#         train_mse_unnorm = 0.0
        
#         for batch in train_loader:
#             # Move data to device
#             for key in batch:
#                 if isinstance(batch[key], torch.Tensor):
#                     batch[key] = batch[key].to(device)
            
#             # Forward pass (with teacher forcing during training)
#             optimizer.zero_grad()
#             predictions = model(batch, teacher_forcing_ratio=teacher_forcing_ratio)
            
#             # Calculate loss
#             loss = criterion(predictions, batch['future'])
            
#             # Calculate unnormalized training MSE
#             pred_unnorm = predictions * batch['scale_position'].view(-1, 1, 1)
#             future_unnorm = batch['future'] * batch['scale_position'].view(-1, 1, 1)
#             train_mse_unnorm += nn.MSELoss()(pred_unnorm, future_unnorm).item()
            
#             # Backward and optimize
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
#             optimizer.step()
            
#             train_loss += loss.item()
        
#         # Calculate average training loss
#         train_loss /= len(train_loader)
#         train_mse_unnorm /= len(train_loader)
        
#         # Validation phase
#         model.eval()
#         val_loss = 0.0
#         val_mae = 0.0
#         val_mse = 0.0
        
#         with torch.no_grad():
#             for batch in val_loader:
#                 # Move data to device
#                 for key in batch:
#                     if isinstance(batch[key], torch.Tensor):
#                         batch[key] = batch[key].to(device)
                
#                 # Forward pass (no teacher forcing during validation)
#                 predictions = model(batch, teacher_forcing_ratio=0.0)
                
#                 # Calculate normalized loss
#                 val_loss += criterion(predictions, batch['future']).item()
                
#                 # Calculate unnormalized metrics
#                 pred_unnorm = predictions * batch['scale_position'].view(-1, 1, 1)
#                 future_unnorm = batch['future'] * batch['scale_position'].view(-1, 1, 1)
                
#                 val_mae += nn.L1Loss()(pred_unnorm, future_unnorm).item()
#                 val_mse += nn.MSELoss()(pred_unnorm, future_unnorm).item()
        
#         # Calculate average validation losses
#         val_loss /= len(val_loader)
#         val_mae /= len(val_loader)
#         val_mse /= len(val_loader)
        
#         # Update learning rate
#         scheduler.step()
        
#         # Update progress bar with metrics
#         progress_bar.set_postfix({
#             'lr': f"{optimizer.param_groups[0]['lr']:.6f}",
#             'train_mse': f"{train_loss:.4f}",
#             'train_mse_unnorm': f"{train_mse_unnorm:.4f}",
#             'val_mse': f"{val_loss:.4f}",
#             'val_mae': f"{val_mae:.4f}",
#             'val_mse_unnorm': f"{val_mse:.4f}"
#         })
        
#         # Save the best model
#         if val_loss < best_val_loss - 1e-3:
#             best_val_loss = val_loss
#             no_improvement = 0
#             torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'val_loss': val_loss,
#                 'val_mae': val_mae,
#                 'val_mse': val_mse
#             }, "best_model.pth")
#         else:
#             no_improvement += 1
#             if no_improvement >= early_stopping_patience:
#                 progress_bar.write("Early stopping!")
#                 break
    
#     # Load the best model
#     checkpoint = torch.load("best_model.pth")
#     model.load_state_dict(checkpoint['model_state_dict'])
    
#     return model, checkpoint

def physics_constraint_loss(predictions, dt=0.1, scale_factor=15.0):
    """
    Add physics-based constraints to ensure realistic trajectories
    
    Args:
        predictions: Trajectory predictions [batch_size, seq_len, output_dim]
        dt: Time step between predictions (in seconds)
        scale_factor: Scale factor to convert normalized coordinates to real-world units
        
    Returns:
        Combined physics loss term
    """
    # Convert to real-world scale for physics calculations (if predictions are normalized)
    # If predictions are already in real-world scale, you can omit this step
    predictions_real = predictions * scale_factor
    
    # Calculate velocities between consecutive points (m/s)
    velocities = (predictions_real[:, 1:] - predictions_real[:, :-1]) / dt
    
    # Calculate accelerations (m/s²)
    accelerations = (velocities[:, 1:] - velocities[:, :-1]) / dt
    
    # Calculate jerk (rate of change of acceleration) (m/s³)
    jerk = (accelerations[:, 1:] - accelerations[:, :-1]) / dt
    
    # Physics constraints with reasonable thresholds for vehicles
    
    # 1. Max acceleration penalty (most vehicles can't exceed ~4 m/s²)
    max_acc = 4.0  # m/s²
    acc_magnitude = torch.norm(accelerations, dim=2)  # Calculate magnitude of acceleration vectors
    acc_penalty = torch.mean(torch.relu(acc_magnitude - max_acc))
    
    # 2. Max velocity penalty (speed limit ~30 m/s or ~110 km/h)
    max_vel = 30.0  # m/s
    vel_magnitude = torch.norm(velocities, dim=2)  # Calculate magnitude of velocity vectors
    vel_penalty = torch.mean(torch.relu(vel_magnitude - max_vel))
    
    # 3. Jerk minimization for smooth trajectories
    jerk_magnitude = torch.norm(jerk, dim=2)
    jerk_penalty = torch.mean(jerk_magnitude)
    
    # 4. Direction consistency (penalize abrupt direction changes)
    # Get normalized velocity vectors
    vel_norm = torch.norm(velocities, dim=2, keepdim=True) + 1e-8  # Avoid division by zero
    vel_unit = velocities / vel_norm
    # Calculate cosine similarity between consecutive velocity vectors (1 = same direction, -1 = opposite)
    vel_cos_sim = torch.sum(vel_unit[:, :-1] * vel_unit[:, 1:], dim=2)
    # Penalize when directions differ significantly (cos_sim < 0.7 is about 45 degrees)
    direction_penalty = torch.mean(torch.relu(0.7 - vel_cos_sim))
    
    # Combined physics loss with weights for each component
    return (0.1 * acc_penalty + 
            0.05 * vel_penalty + 
            0.01 * jerk_penalty + 
            0.1 * direction_penalty)
