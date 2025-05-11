import torch
import numpy as np
from torch.utils.data import Dataset


class ImprovedTrajectoryDataset(Dataset):
    def __init__(self, npz_file_path, split='train', 
                 position_scale=5000.0, velocity_scale=16.0, 
                 robust_norm=True, augment=False):
        """
        Enhanced trajectory dataset with optimized normalization
        
        Args:
            npz_file_path: Path to data file
            split: 'train' or 'test'
            position_scale: Normalization scale for positions (from analysis)
            velocity_scale: Normalization scale for velocities (from analysis)
            robust_norm: Enable robust normalization
            augment: Enable data augmentation for training
        """
        data = np.load(npz_file_path)
        self.data = data['data']
        self.split = split
        self.position_scale = position_scale
        self.velocity_scale = velocity_scale
        self.robust_norm = robust_norm  
        self.augment = augment and split == 'train'
        
        if split == 'train':
            # For training data, separate history and future
            self.history = self.data[..., :50, :]  # First 50 timesteps
            self.future = self.data[:, 0, 50:, :2]  # Future 60 timesteps, focal agent only
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
        
        # Use the last timeframe as the origin
        origin = hist[0, 49, :2].copy()
        
        # Apply robust normalization with separate scales
        if self.robust_norm:
            # Center positions around the origin
            hist[..., :2] = hist[..., :2] - origin
            
            # Check for outliers in positions
            pos_magnitudes = np.sqrt(np.sum(hist[..., :2]**2, axis=-1))
            pos_outlier_mask = pos_magnitudes > 3 * self.position_scale
            
            # Clip outlier positions - add check for non-zero values
            if np.any(pos_outlier_mask):
                # Avoid division by zero by creating a safe version of magnitudes
                safe_magnitudes = np.maximum(pos_magnitudes, 1e-6)  # Add small epsilon
                scale_factors = np.minimum(3 * self.position_scale / safe_magnitudes, 1.0)
                scale_factors = scale_factors.reshape(*scale_factors.shape, 1)
                hist[..., :2] = hist[..., :2] * scale_factors
            
            # Check for outliers in velocities
            vel_magnitudes = np.sqrt(np.sum(hist[..., 2:4]**2, axis=-1))
            vel_outlier_mask = vel_magnitudes > 3 * self.velocity_scale
            
            # Clip outlier velocities - add check for non-zero values
            if np.any(vel_outlier_mask):
                # Avoid division by zero
                safe_vel_magnitudes = np.maximum(vel_magnitudes, 1e-6)  # Add small epsilon
                vel_scale_factors = np.minimum(3 * self.velocity_scale / safe_vel_magnitudes, 1.0)
                vel_scale_factors = vel_scale_factors.reshape(*vel_scale_factors.shape, 1)
                hist[..., 2:4] = hist[..., 2:4] * vel_scale_factors
            
            # Normalize with different scales for positions and velocities
            hist[..., :2] = hist[..., :2] / self.position_scale
            hist[..., 2:4] = hist[..., 2:4] / self.velocity_scale
        else:
            # Basic normalization approach
            hist[..., :2] = hist[..., :2] - origin
            hist[..., :2] = hist[..., :2] / self.position_scale
            hist[..., 2:4] = hist[..., 2:4] / self.velocity_scale
        
        # Create data item
        if self.split == 'train':
            if 'future' not in locals() or future is None:
                future = self.future[idx].copy()
            
            # Normalize future trajectory
            future = future - origin
            future = future / self.position_scale
            
            return {
                'history': torch.tensor(hist, dtype=torch.float32),
                'future': torch.tensor(future, dtype=torch.float32),
                'origin': torch.tensor(origin, dtype=torch.float32),
                'position_scale': torch.tensor(self.position_scale, dtype=torch.float32),
                'velocity_scale': torch.tensor(self.velocity_scale, dtype=torch.float32),
                'scale': torch.tensor(self.position_scale, dtype=torch.float32)  # Add for backward compatibility
            }
        else:
            return {
                'history': torch.tensor(hist, dtype=torch.float32),
                'origin': torch.tensor(origin, dtype=torch.float32),
                'position_scale': torch.tensor(self.position_scale, dtype=torch.float32),
                'velocity_scale': torch.tensor(self.velocity_scale, dtype=torch.float32),
                'scale': torch.tensor(self.position_scale, dtype=torch.float32)  # Add for backward compatibility
            }
    
    def denormalize_positions(self, normalized_positions, origins=None, scales=None):
        """Convert normalized position values back to the original scale"""
        if origins is None:
            # Just use scale if no origin is provided
            return normalized_positions * (scales if scales is not None else self.position_scale)
        else:
            # Use both scale and origin
            if isinstance(normalized_positions, torch.Tensor):
                scales = scales if scales is not None else self.position_scale
                if isinstance(scales, torch.Tensor) and scales.dim() == 1:
                    scales = scales.view(-1, 1, 1)
                return normalized_positions * scales + origins.view(-1, 1, 2)
            else:
                scales = scales if scales is not None else self.position_scale
                if isinstance(scales, np.ndarray) and scales.ndim == 1:
                    scales = scales.reshape(-1, 1, 1)
                return normalized_positions * scales + origins.reshape(-1, 1, 2)