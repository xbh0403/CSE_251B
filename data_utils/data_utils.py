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
            


import torch
import numpy as np
from torch.utils.data import Dataset

class PhysicsTrajectoryDataset(Dataset):
    def __init__(self, npz_file_path, split='train', position_scale=7.0, velocity_scale=5.0, 
                 add_physics_features=True, augment=False):
        data = np.load(npz_file_path)
        self.data = data['data']
        self.split = split
        self.position_scale = position_scale
        self.velocity_scale = velocity_scale
        self.add_physics_features = add_physics_features
        self.augment = augment and split == 'train'
        
        if split == 'train':
            # For training data, separate history and future
            self.history = self.data[..., :50, :]  # First 50 timesteps
            self.future = self.data[:, 0, 50:, :2]  # Future 60 timesteps, only for focal agent
        else:
            # For test data, only have history
            self.history = self.data
    
    def compute_physics_features(self, positions, velocities):
        """Compute physics-based features from positions and velocities"""
        batch_size = positions.shape[0]
        seq_len = positions.shape[1]
        
        # Calculate heading from velocities
        headings = np.arctan2(velocities[..., 1], velocities[..., 0])  # [batch, agents, time]
        
        # Calculate speed (velocity magnitude)
        speeds = np.sqrt(np.sum(velocities**2, axis=-1))  # [batch, agents, time]
        
        # Calculate acceleration (velocity change)
        accelerations = np.zeros_like(velocities)
        accelerations[..., 1:, :] = (velocities[..., 1:, :] - velocities[..., :-1, :]) / 0.1  # 10Hz sampling
        
        # Calculate angular velocity (heading change)
        angular_velocities = np.zeros_like(headings)
        heading_diff = headings[..., 1:] - headings[..., :-1]
        # Normalize angular difference to [-π, π]
        angular_velocities[..., 1:] = np.arctan2(np.sin(heading_diff), np.cos(heading_diff)) / 0.1
        
        # Stack all features
        physics_features = np.stack([
            speeds,
            headings,
            np.linalg.norm(accelerations, axis=-1),  # Acceleration magnitude
            angular_velocities
        ], axis=-1)  # [batch, agents, time, 4]
        
        return physics_features
    
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
        
        # Extract positions and velocities
        positions = hist[..., :2].copy()
        velocities = hist[..., 2:4].copy()
        
        # Use the last timeframe of the historical trajectory as the origin
        origin = positions[0, -1, :].copy()
        
        # Center positions around the origin
        positions = positions - origin
        
        # Add physics features if requested
        if self.add_physics_features:
            physics_feats = self.compute_physics_features(positions, velocities)
            
            # Normalize physics features
            # Speed (already positive)
            physics_feats[..., 0] = np.clip(physics_feats[..., 0] / 10.0, 0, 10)  # Max speed ~100 m/s
            # Heading is already normalized [-π, π]
            # Acceleration magnitude (already positive)
            physics_feats[..., 2] = np.clip(physics_feats[..., 2] / 5.0, 0, 10)  # Max accel ~50 m/s²
            # Angular velocity (can be positive or negative)
            physics_feats[..., 3] = np.clip(physics_feats[..., 3] / 1.0, -5, 5)  # Max ~10 rad/s
        
        # Safe normalize positions and velocities
        positions = positions / self.position_scale
        
        # Safe normalize velocities - avoid divide by zero
        velocities = velocities / self.velocity_scale
        
        # Combine features
        if self.add_physics_features:
            # Combine original features with physics features
            # Original: positions (x,y) and velocities (vx,vy) = 4 features
            # New: positions (2) + velocities (2) + physics (4) = 8 features
            combined_features = np.concatenate([
                positions,
                velocities,
                physics_feats
            ], axis=-1)  # [num_agents, seq_len, 8]
        else:
            combined_features = np.concatenate([positions, velocities], axis=-1)  # [num_agents, seq_len, 4]
        
        # Create data item
        if self.split == 'train':
            if 'future' not in locals() or future is None:
                future = self.future[idx].copy()
            
            # Normalize future trajectory
            future = future - origin
            future = future / self.position_scale
            
            # Get last observed state for physics-based loss
            last_pos = positions[0, -1, :]
            last_vel = velocities[0, -1, :]
            
            return {
                'history': torch.tensor(combined_features, dtype=torch.float32),
                'future': torch.tensor(future, dtype=torch.float32),
                'origin': torch.tensor(origin, dtype=torch.float32),
                'position_scale': torch.tensor(self.position_scale, dtype=torch.float32),
                'velocity_scale': torch.tensor(self.velocity_scale, dtype=torch.float32),
                'last_pos': torch.tensor(last_pos, dtype=torch.float32),
                'last_vel': torch.tensor(last_vel, dtype=torch.float32)
            }
        else:
            # Get last observed state for physics-based prediction
            last_pos = positions[0, -1, :]
            last_vel = velocities[0, -1, :]
            
            return {
                'history': torch.tensor(combined_features, dtype=torch.float32),
                'origin': torch.tensor(origin, dtype=torch.float32),
                'position_scale': torch.tensor(self.position_scale, dtype=torch.float32),
                'velocity_scale': torch.tensor(self.velocity_scale, dtype=torch.float32),
                'last_pos': torch.tensor(last_pos, dtype=torch.float32),
                'last_vel': torch.tensor(last_vel, dtype=torch.float32)
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