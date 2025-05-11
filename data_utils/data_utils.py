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
            self.history = self.data[..., :50, :]  # First 50 timesteps
            self.future = self.data[:, 0, 50:, :2]  # Future 60 timesteps, only for focal agent
        else:
            # For test data, only have history
            self.history = self.data
    
    def __len__(self):
        return self.history.shape[0]
    
    def __getitem__(self, idx):
        # Get history data
        hist = self.history[idx].copy()
        
        # Apply data augmentation for training
        if self.augment:
            future = None
            if np.random.rand() < 0.5:
                theta = np.random.uniform(-np.pi, np.pi)
                R = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]], dtype=np.float32)
                hist[..., :2] = hist[..., :2] @ R
                hist[..., 2:4] = hist[..., 2:4] @ R
                
                if self.split == 'train':
                    future = self.future[idx].copy()
                    future = future @ R
            
            if np.random.rand() < 0.5:
                hist[..., 0] *= -1
                hist[..., 2] *= -1
                
                if self.split == 'train':
                    if future is None:
                        future = self.future[idx].copy()
                    future[:, 0] *= -1
        
        # Use the last timeframe of the historical trajectory as the origin
        origin = hist[0, 49, :2].copy()
        hist[..., :2] = hist[..., :2] - origin
        
        # Normalize the historical trajectory
        hist[..., :4] = hist[..., :4] / self.scale
        
        # Create data item
        if self.split == 'train':
            if 'future' not in locals() or future is None:
                future = self.future[idx].copy()
            future = future - origin
            future = future / self.scale
            
            return {
                'history': torch.tensor(hist, dtype=torch.float32),
                'future': torch.tensor(future, dtype=torch.float32),
                'origin': torch.tensor(origin, dtype=torch.float32),
                'scale': torch.tensor(self.scale, dtype=torch.float32)
            }
        else:
            return {
                'history': torch.tensor(hist, dtype=torch.float32),
                'origin': torch.tensor(origin, dtype=torch.float32),
                'scale': torch.tensor(self.scale, dtype=torch.float32)
            }
    
    def denormalize_positions(self, normalized_positions, origins=None, scales=None):
        """Convert normalized position values back to the original scale"""
        if origins is None:
            # Just use scale if no origin is provided
            if isinstance(normalized_positions, torch.Tensor):
                return normalized_positions * self.scale
            else:
                return normalized_positions * self.scale
        else:
            # Use both scale and origin
            if isinstance(normalized_positions, torch.Tensor):
                return normalized_positions * self.scale + origins
            else:
                return normalized_positions * self.scale + origins