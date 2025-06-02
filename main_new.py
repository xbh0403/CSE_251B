# main.py

import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
import torch.nn as nn

from data_utils.data_utils import TrajectoryDataset
from models.model import Seq2SeqLSTMModel
from models.train import train_model, save_training_metrics
from models.predict import (
    generate_predictions,
    create_submission,
    generate_multimodal_predictions
)

# -----------------------------------------------
# 1. 在此函数中初始化随机种子、路径和超参，然后调用训练、测试等
# -----------------------------------------------
def main():
    # -------------------- 1. 环境与随机种子 --------------------
    torch.manual_seed(42)
    np.random.seed(42)
    
    # -------------------- 2. 路径配置 --------------------
    if os.getcwd().startswith('/tscc'):
        train_path = '/tscc/nfs/home/bax001/scratch/CSE_251B/data/train.npz'
        test_path  = '/tscc/nfs/home/bax001/scratch/CSE_251B/data/test_input.npz'
    else:
        train_path = 'data/train.npz'
        test_path  = 'data/test_input.npz'
    
    # mean_path, std_path 如果你不再使用 global normalization，这里可注释掉
    # mean_path = 'mean_vector.npy'
    # std_path  = 'std_vector.npy'
    
    os.makedirs("logs", exist_ok=True)
    
    # -------------------- 3. 超参数 --------------------
    batch_size = 64
    hidden_dim = 64     # ⚠️ 改为 64，与图示保持一致
    num_layers = 2
    teacher_forcing_ratio = 0.0   # 训练时完全不使用 Teacher Forcing
    num_epochs = 100
    early_stopping_patience = 10
    lr = 1e-3
    weight_decay = 1e-4
    lr_step_size = 20
    lr_gamma = 0.25
    
    # -------------------- 4. 数据集加载 --------------------
    print("Creating datasets...")
    full_train_dataset = TrajectoryDataset(
        npz_file_path=train_path,
        split='train',
        scale_position=7.0,
        scale_heading=1.0,
        scale_velocity=1.0,
        augment=True
    )
    competition_test_dataset = TrajectoryDataset(
        npz_file_path=test_path,
        split='test',
        scale_position=7.0,
        scale_heading=1.0,
        scale_velocity=1.0,
        augment=False
    )
    
    # 按 70/15/15 划分训练/验证/内部测试
    dataset_size = len(full_train_dataset)
    train_size  = int(0.70 * dataset_size)
    val_size    = int(0.15 * dataset_size)
    test_size   = dataset_size - train_size - val_size
    
    train_dataset, val_dataset, internal_test_dataset = random_split(
        full_train_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader    = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  drop_last=True, num_workers=4, pin_memory=True)
    val_loader      = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4, pin_memory=True)
    internal_loader = DataLoader(internal_test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    competition_loader = DataLoader(competition_test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # -------------------- 5. 模型选择 & 初始化 --------------------
    model_type = "lstm"  # 目前仅示例 lstm，如需支持其它类型，可自行扩展
    model_name = f"{model_type}_h{hidden_dim}_l{num_layers}_b{batch_size}"
    
    print(f"Creating {model_type} model...")
    if model_type == "lstm":
        model = Seq2SeqLSTMModel(
            input_dim=6,
            hidden_dim=hidden_dim,
            output_seq_len=60,
            output_dim=2,
            num_layers=num_layers
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    model = model.to(device)
    
    # -------------------- 6. 训练 --------------------
    print(f"Training {model_type} model...")
    model, checkpoint, metrics_history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        early_stopping_patience=early_stopping_patience,
        lr=lr,
        weight_decay=weight_decay,
        lr_step_size=lr_step_size,
        lr_gamma=lr_gamma,
        teacher_forcing_ratio=teacher_forcing_ratio,
        model_name=model_name,
        save_logs=True
    )
    
    print("Best model saved with validation metrics:")
    print(f"  Val Hybrid Loss: {checkpoint['val_loss']:.4f}")
    print(f"  Val MAE (raw):   {checkpoint['val_mae']:.4f}")
    print(f"  Val MSE (raw):   {checkpoint['val_mse']:.4f}")
    
    # -------------------- 7. 在内部测试集上评估 --------------------
    print("Evaluating on internal test set...")
    model.eval()
    test_mae = 0.0
    test_mse = 0.0
    
    all_preds_unnorm = []
    all_gts_unnorm   = []
    all_histories    = []
    
    dataset_obj = full_train_dataset  # 用于调用 denormalize_positions
    
    with torch.no_grad():
        for batch in internal_loader:
            # 把所有 Tensor 移到 GPU/CPU
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
    
            # 预测 [B, 60, 2]（normalized）
            preds_norm = model(batch, teacher_forcing_ratio=0.0)
    
            # 反归一化
            origins = batch['origin']          # [B, 2]
            scale_pos = batch['scale_position'].view(-1, 1)  # [B, 1]
    
            B = preds_norm.size(0)
            for i in range(B):
                pred_norm_i = preds_norm[i]         # [60, 2]
                gt_norm_i   = batch['future'][i]    # [60, 2]
                origin_i    = origins[i]            # [2]
                sp_i        = scale_pos[i]          # [1]
    
                # 反归一化到物理坐标：pred_raw_i, gt_raw_i
                # TrajectoryDataset.denormalize_positions 会执行: normalized * scale + origin
                pred_raw_i = dataset_obj.denormalize_positions(
                    normalized_positions=pred_norm_i.cpu(),
                    origins=origin_i.cpu()
                )  # [60,2] (cpu)
    
                gt_raw_i = dataset_obj.denormalize_positions(
                    normalized_positions=gt_norm_i.cpu(),
                    origins=origin_i.cpu()
                )  # [60,2]
    
                all_preds_unnorm.append(pred_raw_i.numpy())
                all_gts_unnorm.append(gt_raw_i.numpy())
    
                # 记录 focal agent 历史 (raw)：history 里前两列本身已经减去 origin 并除以 scale，
                # 这里直接用 raw = (normalized * scale + origin) 来还原。也可直接从原数据读取 raw。
                hist_norm_i = batch['history'][i, 0, :, :2].cpu()  # [50,2]
                hist_raw_i = hist_norm_i * sp_i + origin_i.cpu()   # [50,2]
                all_histories.append(hist_raw_i.numpy())
    
                # 计算物理尺度下的 MAE/MSE
                pred_raw_tensor = torch.from_numpy(pred_raw_i).to(device)
                gt_raw_tensor   = torch.from_numpy(gt_raw_i).to(device)
                test_mae += nn.L1Loss()(pred_raw_tensor, gt_raw_tensor).item()
                test_mse += nn.MSELoss()(pred_raw_tensor, gt_raw_tensor).item()
    
    total_samples = len(internal_loader.dataset)
    test_mae /= total_samples
    test_mse /= total_samples
    
    # 保存到 CSV
    test_log_path = os.path.join("logs", "test_metrics.csv")
    test_metrics = {
        'model': model_name,
        'test_mae': test_mae,
        'test_mse': test_mse
    }
    test_df = pd.DataFrame([test_metrics])
    if os.path.exists(test_log_path):
        test_df.to_csv(test_log_path, mode='a', header=False, index=False)
    else:
        test_df.to_csv(test_log_path, index=False)
    
    print("Internal test results:")
    print(f"  MAE: {test_mae:.4f}")
    print(f"  MSE: {test_mse:.4f}")
    print(f"Saved test metrics to {test_log_path}")
    
    # -------------------- 8. 可视化内部测试样本 --------------------
    all_preds_unnorm = np.stack(all_preds_unnorm, axis=0)  # [N_int, 60, 2]
    all_gts_unnorm   = np.stack(all_gts_unnorm,   axis=0)  # [N_int, 60, 2]
    all_histories    = np.stack(all_histories,    axis=0)  # [N_int, 50, 2]
    
    viz_dir = os.path.join("logs", f"{model_name}_test_viz")
    os.makedirs(viz_dir, exist_ok=True)
    
    # 随机选 5 条做可视化
    num_viz = min(5, all_preds_unnorm.shape[0])
    indices = np.random.choice(all_preds_unnorm.shape[0], num_viz, replace=False)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for i, idx in enumerate(indices):
        ax = axes[i]
        hist_xy = all_histories[idx]    # [50,2]
        pred_xy = all_preds_unnorm[idx] # [60,2]
        gt_xy   = all_gts_unnorm[idx]   # [60,2]
    
        ax.plot(hist_xy[:, 0], hist_xy[:, 1], 'k.-', label='History')
        ax.plot(pred_xy[:, 0], pred_xy[:, 1], 'b.-', label='Prediction')
        ax.plot(gt_xy[:, 0], gt_xy[:, 1], 'g.-', label='Ground Truth')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"Example {i+1}")
        ax.grid(True)
        if i == 0:
            ax.legend()
    
    # 最后一个子图显示指标
    ax = axes[5]
    ax.axis('off')
    metrics_text = (
        f"Test Metrics:\n"
        f"MAE: {test_mae:.4f}\n"
        f"MSE: {test_mse:.4f}\n\n"
        f"Best Val Metrics:\n"
        f"Val MAE: {checkpoint['val_mae']:.4f}\n"
        f"Val MSE: {checkpoint['val_mse']:.4f}\n"
    )
    ax.text(0.1, 0.5, metrics_text, fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "test_examples.png"), dpi=300)
    plt.close(fig)
    print(f"Saved test visualizations to {viz_dir}")
    
    # -------------------- 9. 生成竞赛/最终测试预测 --------------------
    print("Generating predictions for competition test data...")
    model.eval()
    if hasattr(model, 'num_modes'):
        comp_preds = generate_multimodal_predictions(model, competition_loader)
    else:
        comp_preds = generate_predictions(model, competition_loader)
    
    submission_path = f"{model_name}_submission.csv"
    print("Creating submission file at:", submission_path)
    create_submission(comp_preds, output_file=submission_path)
    print("Submission saved.")
    
    print("All done.")


if __name__ == "__main__":
    main()
