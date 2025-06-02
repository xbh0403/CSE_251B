# models/train.py

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


def save_training_metrics(metrics_history: dict, model_name: str, save_dir: str):
    """
    将训练过程中记录的 metrics_history 保存为 CSV、JSON，并绘制曲线图。

    metrics_history 字典的键（示例）:
        - 'epoch'
        - 'learning_rate'
        - 'train_hybrid_loss'
        - 'train_mse_unnorm'
        - 'val_hybrid_loss'
        - 'val_mae_unnorm'
        - 'val_mse_unnorm'

    Args:
        metrics_history: dict，包含上述字段
        model_name: str，用于文件名前缀
        save_dir: str，保存目录

    Returns:
        csv_path: str，保存的 CSV 路径
        json_path: str，保存的 JSON 路径
        viz_dir: str，保存图像的子目录
    """
    os.makedirs(save_dir, exist_ok=True)

    # 转成 DataFrame
    metrics_df = pd.DataFrame(metrics_history)

    # 保存 CSV
    csv_path = os.path.join(save_dir, f"{model_name}_metrics.csv")
    metrics_df.to_csv(csv_path, index=False)

    # 保存 JSON
    json_path = os.path.join(save_dir, f"{model_name}_metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics_history, f, indent=2)

    # 新建子目录存放可视化图
    viz_dir = os.path.join(save_dir, "plots")
    os.makedirs(viz_dir, exist_ok=True)

    # —— 绘制曲线图 —— #

    # 1) Hybrid Loss: Training vs Validation
    plt.figure()
    plt.plot(metrics_df["train_hybrid_loss"], label="Train Hybrid Loss")
    plt.plot(metrics_df["val_hybrid_loss"], label="Val Hybrid Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Hybrid Loss (MSE + α·MAE)")
    plt.title("Hybrid Loss per Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "hybrid_loss.png"))
    plt.close()

    # 2) Unnormalized MSE: Training vs Validation
    plt.figure()
    plt.plot(metrics_df["train_mse_unnorm"], label="Train MSE (raw)")
    plt.plot(metrics_df["val_mse_unnorm"], label="Val MSE (raw)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE (physical scale)")
    plt.title("Unnormalized MSE per Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "mse_unnorm.png"))
    plt.close()

    # 3) Unnormalized MAE (仅 Validation)
    plt.figure()
    plt.plot(metrics_df["val_mae_unnorm"], label="Val MAE (raw)")
    plt.xlabel("Epoch")
    plt.ylabel("MAE (physical scale)")
    plt.title("Unnormalized MAE (Validation) per Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "mae_unnorm.png"))
    plt.close()

    return csv_path, json_path, viz_dir


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    num_epochs: int = 100,
    early_stopping_patience: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    lr_step_size: int = 20,
    lr_gamma: float = 0.25,
    teacher_forcing_ratio: float = 0.0,
    model_name: str = "seq2seq_model",
    save_logs: bool = True
):
    """
    使用“混合损失 (MSE + α·MAE)”训练模型，同时计算物理尺度下的 MSE/MAE，用于评估。
    1) 在归一化空间内计算 hybrid_loss = MSE(pred_norm, gt_norm) + α·MAE(pred_norm, gt_norm)
    2) 用 scale_position 把 pred_norm → pred_raw（物理坐标），再计算 unnorm MSE/MAE
    3) Early‐stopping based on hybrid_loss
    """
    device = next(model.parameters()).device

    # 定义损失
    mse_criterion = nn.MSELoss(reduction="mean")
    mae_criterion = nn.L1Loss(reduction="mean")
    alpha = 0.5  # 混合损失系数，可根据实验调整

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=lr_step_size, gamma=lr_gamma
    )

    # 用来记录每个 epoch 的指标
    metrics_history = {
        "epoch": [],
        "learning_rate": [],
        "train_hybrid_loss": [],
        "train_mse_unnorm": [],
        "val_hybrid_loss": [],
        "val_mae_unnorm": [],
        "val_mse_unnorm": []
    }

    best_val_loss = float("inf")
    no_improvement = 0

    progress_bar = tqdm(range(num_epochs), desc="Epoch", unit="epoch")
    for epoch in progress_bar:
        # ------- 训练阶段 ------- #
        model.train()
        running_hybrid_loss = 0.0
        running_mse_unnorm = 0.0

        for batch in train_loader:
            # 将所有 tensor 移到 device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            optimizer.zero_grad()
            # 得到 normalized 空间的预测: shape [B,60,2]
            preds_norm = model(batch, teacher_forcing_ratio=teacher_forcing_ratio)

            # 1) 归一化空间下的混合损失
            mse_loss = mse_criterion(preds_norm, batch["future"])
            mae_loss = mae_criterion(preds_norm, batch["future"])
            hybrid_loss = mse_loss + alpha * mae_loss
            running_hybrid_loss += hybrid_loss.item()

            # 2) 物理空间下的 unnorm MSE/MAE
            #    scale_position: Tensor [B] 或 [B,1] → reshape 成 [B,1,1]
            scale_pos = batch["scale_position"].view(-1, 1, 1)
            preds_raw = preds_norm * scale_pos       # [B,60,2]
            gt_raw    = batch["future"] * scale_pos  # [B,60,2]

            mse_phys = mse_criterion(preds_raw, gt_raw)
            running_mse_unnorm += mse_phys.item()

            # 3) 反向传播
            hybrid_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        avg_train_hybrid = running_hybrid_loss / len(train_loader)
        avg_train_mse_unnorm = running_mse_unnorm / len(train_loader)

        # ------- 验证阶段 ------- #
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

                mse_loss = mse_criterion(preds_norm, batch["future"])
                mae_loss = mae_criterion(preds_norm, batch["future"])
                hybrid_loss = mse_loss + alpha * mae_loss
                running_val_hybrid += hybrid_loss.item()

                scale_pos = batch["scale_position"].view(-1, 1, 1)
                preds_raw = preds_norm * scale_pos
                gt_raw    = batch["future"] * scale_pos

                val_mae_phys = mae_criterion(preds_raw, gt_raw)
                val_mse_phys = mse_criterion(preds_raw, gt_raw)
                running_val_mae_unnorm += val_mae_phys.item()
                running_val_mse_unnorm += val_mse_phys.item()

        avg_val_hybrid = running_val_hybrid / len(val_loader)
        avg_val_mae_unnorm = running_val_mae_unnorm / len(val_loader)
        avg_val_mse_unnorm = running_val_mse_unnorm / len(val_loader)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # 记录到 metrics_history
        metrics_history["epoch"].append(epoch)
        metrics_history["learning_rate"].append(current_lr)
        metrics_history["train_hybrid_loss"].append(avg_train_hybrid)
        metrics_history["train_mse_unnorm"].append(avg_train_mse_unnorm)
        metrics_history["val_hybrid_loss"].append(avg_val_hybrid)
        metrics_history["val_mae_unnorm"].append(avg_val_mae_unnorm)
        metrics_history["val_mse_unnorm"].append(avg_val_mse_unnorm)

        progress_bar.set_postfix({
            "lr":                 f"{current_lr:.6f}",
            "train_hybrid_loss":  f"{avg_train_hybrid:.4f}",
            "train_mse_unnorm":   f"{avg_train_mse_unnorm:.4f}",
            "val_hybrid_loss":    f"{avg_val_hybrid:.4f}",
            "val_mae_unnorm":     f"{avg_val_mae_unnorm:.4f}"
        })

        # 保存最佳模型 (根据 val_hybrid_loss)
        if avg_val_hybrid < best_val_loss - 1e-3:
            best_val_loss = avg_val_hybrid
            no_improvement = 0
            checkpoint_info = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avg_val_hybrid,
                "val_mae": avg_val_mae_unnorm,
                "val_mse": avg_val_mse_unnorm,
                "metrics_history": metrics_history
            }
            torch.save(checkpoint_info, "best_model.pth")
        else:
            no_improvement += 1
            if no_improvement >= early_stopping_patience:
                progress_bar.write("Early stopping!")
                break

    # 如果需要保存日志 (CSV/JSON/plots)
    if save_logs:
        csv_path, json_path, viz_dir = save_training_metrics(
            metrics_history,
            model_name=model_name,
            save_dir="logs"
        )
        progress_bar.write(f"Training metrics saved to {csv_path} and {json_path}")
        progress_bar.write(f"Visualization plots saved to {viz_dir}")

    # 加载最佳模型权重
    checkpoint = torch.load("best_model.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    return model, checkpoint, metrics_history
