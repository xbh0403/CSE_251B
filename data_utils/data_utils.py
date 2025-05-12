import torch
import numpy as np
from torch.utils.data import Dataset

class TrajectoryDataset(Dataset):
    def __init__(self, npz_file_path, split='train', scale=7.0, augment=False):
        data = np.load(npz_file_path)
        self.data = data['data']
        self.split = split
        self.scale = scale
        self.augment = augment and split == 'train'
        
        if split == 'train':
            # For training data, separate history and future
            self.history = self.data[..., :50, :]   # (num_scenes, num_agents, T_hist, features)
            self.future = self.data[:, 0, 50:, :2]  # (num_scenes, T_fut, 2)
        else:
            # For test data, only have history
            self.history = self.data
    
    def __len__(self):
        return self.history.shape[0]
    
    def __getitem__(self, idx):
        # Raw history for mask computation
        raw_hist = self.history[idx].copy()  # shape: (num_agents, T_hist, features)
        # Create mask: 1 where any feature is non-zero, else 0
        mask = (np.abs(raw_hist).sum(axis=-1) > 0).astype(np.float32)  # (num_agents, T_hist)
        
        # Work on a separate array for augmentation/normalization
        hist = raw_hist
        future = None

        # Data augmentation (train only)
        if self.augment:
            if np.random.rand() < 0.5:
                theta = np.random.uniform(-np.pi, np.pi)
                R = np.array([[np.cos(theta), -np.sin(theta)],
                              [np.sin(theta),  np.cos(theta)]], dtype=np.float32)
                # rotate x,y in hist
                hist[..., :2] = hist[..., :2] @ R
                # rotate velocities if present
                hist[..., 2:4] = hist[..., 2:4] @ R
                # rotate future
                if self.split == 'train':
                    future = self.future[idx].copy() @ R

            if np.random.rand() < 0.5:
                hist[..., 0] *= -1
                hist[..., 2] *= -1
                if self.split == 'train':
                    if future is None:
                        future = self.future[idx].copy()
                    future[:, 0] *= -1

        # Use the last history point of focal agent as new origin
        origin = hist[0, 49, :2].copy()
        # shift all positions
        hist[..., :2] -= origin
        # normalize
        hist[..., :4] /= self.scale

        if self.split == 'train':
            if future is None:
                future = self.future[idx].copy()
            future = future - origin
            future = future / self.scale

            return {
                'history': torch.tensor(hist, dtype=torch.float32),
                'mask':    torch.tensor(mask, dtype=torch.float32),
                'future':  torch.tensor(future, dtype=torch.float32),
                'origin':  torch.tensor(origin, dtype=torch.float32),
                'scale':   torch.tensor(self.scale, dtype=torch.float32)
            }
        else:
            return {
                'history': torch.tensor(hist, dtype=torch.float32),
                'mask':    torch.tensor(mask, dtype=torch.float32),
                'origin':  torch.tensor(origin, dtype=torch.float32),
                'scale':   torch.tensor(self.scale, dtype=torch.float32)
            }
    
    def denormalize_positions(self, normalized_positions, origins=None, scales=None):
        """Convert normalized position values back to the original scale"""
        if origins is None:
            return normalized_positions * self.scale
        else:
            return normalized_positions * self.scale + origins
