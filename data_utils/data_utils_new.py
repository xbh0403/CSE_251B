import os
import torch
import numpy as np
from torch.utils.data import Dataset

class TrajectoryDataset(Dataset):
    """
    改进版 TrajectoryDataset，增加了：
      1. 全局 z-score 归一化（可选 mean/std 输入）
      2. 更多时序数据增强：随机缩放、噪声、随机丢帧
      3. 位置/速度/朝向基于全局 mean/std 或原先的 scale 常数
      4. 支持 train/test 模式，train 模式下按 70/15/15 划分时可做样本平衡
    """

    def __init__(
        self,
        npz_file_path: str,
        split: str = 'train',
        mean_path: str = None,
        std_path: str = None,
        scale_position: float = 7.0,
        scale_heading: float = 1.0,
        scale_velocity: float = 1.0,
        augment: bool = False
    ):
        """
        Args:
            npz_file_path: 包含 key 'data' 的 .npz 文件路径，data 形状为 [N, num_agents, seq_len+future_len, feature_dim]
            split: 'train' 或 'test'
            mean_path: 全局 mean 的 .npy 文件路径（shape 应为 [feature_dim]）
            std_path: 全局 std 的 .npy 文件路径（shape 应为 [feature_dim]）
            scale_position / scale_heading / scale_velocity: 如果未提供 mean/std，仍可按常数归一化
            augment: 仅在 split='train' 时生效，是否启用数据增强
        """
        # 加载原始数据
        data = np.load(npz_file_path)
        self.data = data['data']  # 形状 [N, num_agents, T, feature_dim]
        self.split = split
        self.augment = augment and split == 'train'

        # 读取全局 mean/std（可选）
        if mean_path is not None and std_path is not None:
            self.global_mean = np.load(mean_path).astype(np.float32)  # shape [feature_dim]
            self.global_std  = np.load(std_path).astype(np.float32)
            assert self.global_mean.shape == self.global_std.shape == (self.data.shape[-1],)
            self.use_zscore = True
        else:
            self.global_mean = None
            self.global_std  = None
            self.use_zscore = False

        # 传统按常数 scale
        self.scale_position = scale_position
        self.scale_heading  = scale_heading
        self.scale_velocity = scale_velocity

        # 如果是训练集，拆分历史和未来
        if split == 'train':
            # 历史长度 50，未来长度 60（focal agent 前两维是位置）
            self.history = self.data[..., :50, :].copy()          # [N, num_agents, 50, feature_dim]
            self.future  = self.data[:, 0, 50:, :2].copy()        # [N, 60, 2]
        else:
            # 测试集只需要历史
            self.history = self.data[..., :50, :].copy()

        # 用于样本平衡：计算各条轨迹的 speed 或转弯比例，后续可在 sampler 中使用
        # 这里只示例计算每个样本的平均速度，用于 histogram-based 采样
        if split == 'train':
            speeds = []
            for i in range(self.history.shape[0]):
                hist_i = self.history[i, 0, :, :2]  # focal agent 50 帧位置 [50,2]
                # 大致估算周跳距离均值
                diffs = np.linalg.norm(np.diff(hist_i, axis=0), axis=1)  # [49,]
                speeds.append(diffs.mean())
            self.speeds = np.array(speeds, dtype=np.float32)  # shape [N,]
        else:
            self.speeds = None

    def __len__(self):
        return self.history.shape[0]

    def __getitem__(self, idx):
        """
        返回一个字典，包含：
            'history': Tensor [num_agents, 50, feature_dim_normed]
            'future' (若 split=='train'): Tensor [60, 2_normed]
            'origin': Tensor [2]
            'scale_position': 标量 Tensor
            'scale_heading': 标量 Tensor
            'scale_velocity': 标量 Tensor
        """
        # 原始历史
        hist = self.history[idx].copy()  # [num_agents, 50, feature_dim]
        future = None

        # ------------------ 1. 全局 z-score 归一化 (如果提供 mean/std) ------------------
        if self.use_zscore:
            # 对整段 hist 先应用 (x - mean) / std
            # mean/std shape = [feature_dim]
            hist = (hist - self.global_mean) / (self.global_std + 1e-6)
            if self.split == 'train':
                future_raw = self.future[idx].copy()  # [60,2]
                # 对 future 里的前两维也做相同归一化：它对应 data[:, :, :2]
                # 但 global_mean/std 前两维即 positions 的 mean/std
                future = (future_raw - self.global_mean[:2]) / (self.global_std[:2] + 1e-6)
        else:
            # 不做 z-score，全程之后再做按常数缩放
            pass

        # ------------------ 2. 数据增强 (仅 train) ------------------
        if self.augment:
            # (1) 随机旋转 ±π
            if np.random.rand() < 0.5:
                theta = np.random.uniform(-np.pi, np.pi)
                R = np.array([
                    [np.cos(theta), -np.sin(theta)],
                    [np.sin(theta),  np.cos(theta)]
                ], dtype=np.float32)
                # 针对 z-score 归一化后的位置/速度 (hist[..., :4]) 进行旋转
                hist[..., :2] = hist[..., :2] @ R          # 位置
                hist[..., 2:4] = hist[..., 2:4] @ R        # 速度或朝向向量
                if self.split == 'train':
                    if future is None:
                        future = self.future[idx].copy()  # raw/scale ize
                        future = future / self.scale_position
                    # future 已经归一化到 normalized space (如果 use_zscore=True 就是 zscore 过)
                    future[..., :2] = future[..., :2] @ R

            # (2) 随机左右翻转
            if np.random.rand() < 0.5:
                hist[..., 0] *= -1
                hist[..., 2] *= -1
                if self.split == 'train':
                    if future is None:
                        future = self.future[idx].copy()
                        future = future / self.scale_position
                    future[..., 0] *= -1

            # (3) 随机缩放（放大/缩小 0.9~1.1）
            if np.random.rand() < 0.5:
                scale_factor = np.random.uniform(0.9, 1.1)
                hist[..., :2] *= scale_factor
                hist[..., 2:4] *= scale_factor
                if self.split == 'train':
                    if future is None:
                        future = self.future[idx].copy()
                        future = future / self.scale_position
                    future[..., :2] *= scale_factor

            # (4) 添加高斯噪声：位置/速度/朝向 各加 small Gaussian noise
            if np.random.rand() < 0.5:
                noise_pos = np.random.normal(0, 0.01, size=hist[..., :2].shape).astype(np.float32)
                noise_vel = np.random.normal(0, 0.01, size=hist[..., 2:4].shape).astype(np.float32)
                noise_head = np.random.normal(0, 0.01, size=hist[..., 4:6].shape).astype(np.float32)
                hist[..., :2]   += noise_pos
                hist[..., 2:4]  += noise_vel
                hist[..., 4:6]  += noise_head
                if self.split == 'train':
                    if future is None:
                        future = self.future[idx].copy()
                        future = future / self.scale_position
                    noise_future = np.random.normal(0, 0.01, size=future.shape).astype(np.float32)
                    future[..., :2] += noise_future

            # (5) 随机丢帧：以 10% 概率把历史中某些时刻置为前一时刻的值
            if np.random.rand() < 0.5:
                # 对 focal agent 以及其它所有 agent 均可丢帧
                for t in range(1, 50):
                    if np.random.rand() < 0.1:
                        hist[:, t, :] = hist[:, t - 1, :].copy()
                if self.split == 'train':
                    # 不对 future 丢帧，只增强历史
                    pass

        # ------------------ 3. 以最后时刻作为 origin，变换到相对坐标 ------------------
        # 注意，此时如果 use_zscore=True，hist[..., :2] 已在 z-score 空间
        # 但对相对运动而言，我们仍以 (raw_position - origin_raw) / std_position
        # 因此需要先把 z-score 空间变回原始物理再做 origin 运算，或者直接在 z-score 空间做 origin。
        # 为保持一致，这里直接在 normalized 空间做 origin 操作。
        origin = hist[0, 49, :2].copy()  # 取 focal agent 最后一帧位置/velocity/heading 已归一化后的值
        # 相对位移 (normalized)
        hist[..., :2] = hist[..., :2] - origin
        if self.split == 'train' and future is not None:
            future = future - origin

        # ------------------ 4. 如果未使用 z-score，则按常数分量归一化 ------------------
        if not self.use_zscore:
            # raw hist: hist[..., :2] 已是 raw space，需先 origin，然后除以 scale
            hist[..., :2] = hist[..., :2] / self.scale_position
            hist[..., 2:4] = hist[..., 2:4] / self.scale_velocity
            hist[..., 4:6] = hist[..., 4:6] / self.scale_heading
            if self.split == 'train':
                if future is None:
                    future = self.future[idx].copy()
                future = future - (self.history[idx, 0, 49, :2].copy())
                future = future / self.scale_position

        # ------------------ 5. 构造输出 ------------------
        # 把 numpy → torch.Tensor
        hist_tensor = torch.tensor(hist, dtype=torch.float32)  # [num_agents, 50, feature_dim_normed]
        origin_tensor = torch.tensor(origin, dtype=torch.float32)  # [2]

        if self.split == 'train':
            future_tensor = torch.tensor(future, dtype=torch.float32)  # [60,2_normed]
            return {
                'history': hist_tensor,
                'future': future_tensor,
                'origin': origin_tensor,
                'scale_position': torch.tensor(self.scale_position, dtype=torch.float32),
                'scale_heading': torch.tensor(self.scale_heading, dtype=torch.float32),
                'scale_velocity': torch.tensor(self.scale_velocity, dtype=torch.float32)
            }
        else:
            return {
                'history': hist_tensor,
                'origin': origin_tensor,
                'scale_position': torch.tensor(self.scale_position, dtype=torch.float32),
                'scale_heading': torch.tensor(self.scale_heading, dtype=torch.float32),
                'scale_velocity': torch.tensor(self.scale_velocity, dtype=torch.float32)
            }

    def denormalize_positions(self, normalized_positions, origins=None, scales=None):
        """
        将归一化（z-score 或除以常数）后的坐标还原到原始物理尺度。
        如果 use_zscore=True，则 normalized_positions 应为 (raw - mean) / std，
        那么直接返回：
            raw = normalized_positions * std[:2] + mean[:2]
        如果 use_zscore=False，则 normalized_positions 应为 (raw - origin) / scale_position，
        那么返回：
            raw = normalized_positions * scale_position + origin
        """
        if self.use_zscore and self.global_mean is not None:
            mean_pos = self.global_mean[:2]
            std_pos  = self.global_std[:2]
            return normalized_positions * std_pos + mean_pos
        else:
            return normalized_positions * self.scale_position + origins

    def denormalize_headings(self, normalized_headings, origins=None, scales=None):
        """
        同理还原朝向（heading）。
        如果 use_zscore=True，则：
            raw_heading = normalized_heading * std[4:6] + mean[4:6]
        否则：
            raw_heading = normalized_heading * scale_heading
        """
        if self.use_zscore and self.global_mean is not None:
            mean_head = self.global_mean[4:6]
            std_head  = self.global_std[4:6]
            return normalized_headings * std_head + mean_head
        else:
            return normalized_headings * self.scale_heading

    def denormalize_velocities(self, normalized_velocities, origins=None, scales=None):
        """
        还原速度（velocity）。如果 use_zscore=True，则：
            raw_vel = normalized_vel * std[2:4] + mean[2:4]
        否则：
            raw_vel = normalized_vel * scale_velocity
        """
        if self.use_zscore and self.global_mean is not None:
            mean_vel = self.global_mean[2:4]
            std_vel  = self.global_std[2:4]
            return normalized_velocities * std_vel + mean_vel
        else:
            return normalized_velocities * self.scale_velocity
