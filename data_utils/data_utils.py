import torch
import numpy as np
from torch.utils.data import Dataset

class TrajectoryDataset(Dataset):
    def __init__(self, npz_file_path, split='train', scale_position=7.0, scale_heading=1.0, scale_velocity=1.0, augment=False):
        data = np.load(npz_file_path)
        self.data = data['data']
        self.split = split
        self.scale_position = scale_position
        self.scale_heading = scale_heading
        self.scale_velocity = scale_velocity
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
        hist[..., :2] = hist[..., :2] / self.scale_position
        hist[..., 2:4] = hist[..., 2:4] / self.scale_velocity
        hist[..., 4:6] = hist[..., 4:6] / self.scale_heading
        
        # Create data item
        if self.split == 'train':
            if 'future' not in locals() or future is None:
                future = self.future[idx].copy()
            future = future - origin
            future = future / self.scale_position

            
            return {
                'history': torch.tensor(hist, dtype=torch.float32),
                'future': torch.tensor(future, dtype=torch.float32),
                'origin': torch.tensor(origin, dtype=torch.float32),
                'scale_position': torch.tensor(self.scale_position, dtype=torch.float32),
                'scale_heading': torch.tensor(self.scale_heading, dtype=torch.float32),
                'scale_velocity': torch.tensor(self.scale_velocity, dtype=torch.float32)
            }
        else:
            return {
                'history': torch.tensor(hist, dtype=torch.float32),
                'origin': torch.tensor(origin, dtype=torch.float32),
                'scale_position': torch.tensor(self.scale_position, dtype=torch.float32),
                'scale_heading': torch.tensor(self.scale_heading, dtype=torch.float32),
                'scale_velocity': torch.tensor(self.scale_velocity, dtype=torch.float32)
            }
    
    def denormalize_positions(self, normalized_positions, origins=None, scales=None):
        """Convert normalized position values back to the original scale"""
        return normalized_positions * self.scale_position + origins
            
    def denormalize_headings(self, normalized_headings, origins=None, scales=None):
        """Convert normalized heading values back to the original scale"""
        return normalized_headings * self.scale_heading
            
    def denormalize_velocities(self, normalized_velocities, origins=None, scales=None):
        return normalized_velocities * self.scale_velocity