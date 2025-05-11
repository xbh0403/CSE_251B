import torch
import numpy as np
from torch.utils.data import Dataset

class TrajectoryDataset(Dataset):
    def __init__(self, npz_file_path, split='train', scale=7.0, augment=False,
                 position_scale=7.0, velocity_scale=5.0, robust_norm=True):
        data = np.load(npz_file_path)
        self.data = data['data']
        self.split = split
        self.scale = scale  # Keep original scale for compatibility
        self.position_scale = position_scale  # Separate scale for positions
        self.velocity_scale = velocity_scale  # Separate scale for velocities
        self.robust_norm = robust_norm  # Enable robust normalization
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
        
        # Apply data augmentation for training (unchanged)
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
        
        # Apply robust normalization if enabled
        if self.robust_norm:
            # Center positions around the origin
            hist[..., :2] = hist[..., :2] - origin
            
            # Check for outliers in positions
            pos_magnitudes = np.sqrt(np.sum(hist[..., :2]**2, axis=-1))
            pos_outlier_mask = pos_magnitudes > 3 * self.position_scale
            
            # Clip outlier positions to maintain reasonable scale
            if np.any(pos_outlier_mask):
                scale_factors = np.minimum(3 * self.position_scale / pos_magnitudes, 1.0)
                scale_factors = scale_factors.reshape(*scale_factors.shape, 1)
                hist[..., :2] = hist[..., :2] * scale_factors
            
            # Check for outliers in velocities
            vel_magnitudes = np.sqrt(np.sum(hist[..., 2:4]**2, axis=-1))
            vel_outlier_mask = vel_magnitudes > 3 * self.velocity_scale
            
            # Clip outlier velocities to maintain reasonable scale
            if np.any(vel_outlier_mask):
                vel_scale_factors = np.minimum(3 * self.velocity_scale / vel_magnitudes, 1.0)
                vel_scale_factors = vel_scale_factors.reshape(*vel_scale_factors.shape, 1)
                hist[..., 2:4] = hist[..., 2:4] * vel_scale_factors
            
            # Normalize with different scales for positions and velocities
            hist[..., :2] = hist[..., :2] / self.position_scale
            hist[..., 2:4] = hist[..., 2:4] / self.velocity_scale
        else:
            # Original normalization approach
            hist[..., :2] = hist[..., :2] - origin
            hist[..., :4] = hist[..., :4] / self.scale
        
        # Create data item
        if self.split == 'train':
            if 'future' not in locals() or future is None:
                future = self.future[idx].copy()
            
            # Normalize future trajectory
            future = future - origin
            if self.robust_norm:
                future = future / self.position_scale
            else:
                future = future / self.scale
            
            return {
                'history': torch.tensor(hist, dtype=torch.float32),
                'future': torch.tensor(future, dtype=torch.float32),
                'origin': torch.tensor(origin, dtype=torch.float32),
                'scale': torch.tensor(self.position_scale if self.robust_norm else self.scale, dtype=torch.float32),
                'velocity_scale': torch.tensor(self.velocity_scale, dtype=torch.float32) if self.robust_norm else None
            }
        else:
            return {
                'history': torch.tensor(hist, dtype=torch.float32),
                'origin': torch.tensor(origin, dtype=torch.float32),
                'scale': torch.tensor(self.position_scale if self.robust_norm else self.scale, dtype=torch.float32),
                'velocity_scale': torch.tensor(self.velocity_scale, dtype=torch.float32) if self.robust_norm else None
            }
    
    def denormalize_positions(self, normalized_positions, origins=None, scales=None):
        """Convert normalized position values back to the original scale"""
        if origins is None:
            # Just use scale if no origin is provided
            if isinstance(normalized_positions, torch.Tensor):
                return normalized_positions * (scales if scales is not None else self.position_scale)
            else:
                return normalized_positions * (scales if scales is not None else self.position_scale)
        else:
            # Use both scale and origin
            if isinstance(normalized_positions, torch.Tensor):
                scales = scales if scales is not None else self.position_scale
                if scales.dim() == 1:
                    scales = scales.view(-1, 1, 1)
                return normalized_positions * scales + origins.view(-1, 1, 2)
            else:
                scales = scales if scales is not None else self.position_scale
                if np.ndim(scales) == 1:
                    scales = scales.reshape(-1, 1, 1)
                return normalized_positions * scales + origins.reshape(-1, 1, 2)