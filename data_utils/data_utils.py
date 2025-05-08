import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class TrajectoryDataset(Dataset):
    def __init__(self, npz_file_path, split='train'):
        data = np.load(npz_file_path)
        self.data = data['data']
        self.split = split
        
        # Extract agent types if available in the data
        self.has_agent_types = 'agent_types' in data
        self.agent_types = data.get('agent_types', None)
        self.agent_type_mapping = {
            'vehicle': 0,
            'pedestrian': 1,
            'motorcyclist': 2,
            'cyclist': 3,
            'bus': 4,
            'static': 5,
            'background': 6,
            'construction': 7,
            'riderless_bicycle': 8,
            'unknown': 9
        }
        
        # Normalize coordinates to improve training stability
        if split == 'train':
            # For training data, separate history and future
            self.history = self.data[..., :50, :]  # First 50 timesteps
            self.future = self.data[:, 0, 50:, :2]  # Future 60 timesteps, only for focal agent
            
            # Calculate normalization statistics (mean, std) from training data
            self.pos_mean = np.mean(self.history[..., :2], axis=(0, 1, 2))
            self.pos_std = np.std(self.history[..., :2], axis=(0, 1, 2))
            self.vel_mean = np.mean(self.history[..., 2:4], axis=(0, 1, 2))
            self.vel_std = np.std(self.history[..., 2:4], axis=(0, 1, 2))
            
            # Save statistics to a file for test-time normalization
            np.savez('normalization_stats.npz', 
                    pos_mean=self.pos_mean, pos_std=self.pos_std,
                    vel_mean=self.vel_mean, vel_std=self.vel_std)
        else:
            # For test data, only have history
            self.history = self.data
            
            # Load normalization statistics
            stats = np.load('normalization_stats.npz')
            self.pos_mean = stats['pos_mean']
            self.pos_std = stats['pos_std']
            self.vel_mean = stats['vel_mean']
            self.vel_std = stats['vel_std']
        
        # Apply normalization
        self.history_normalized = self.history.copy()
        self.history_normalized[..., :2] = (self.history[..., :2] - self.pos_mean) / self.pos_std
        self.history_normalized[..., 2:4] = (self.history[..., 2:4] - self.vel_mean) / self.vel_std
        
        if split == 'train':
            self.future_normalized = (self.future - self.pos_mean) / self.pos_std
    
    def __len__(self):
        return self.history.shape[0]
    
    def __getitem__(self, idx):
        # Convert to tensors
        history = torch.tensor(self.history_normalized[idx], dtype=torch.float32)
        
        # Get agent types if available
        if self.has_agent_types:
            agent_types = torch.tensor(self.agent_types[idx], dtype=torch.long)
        else:
            # If agent types not provided, use default type (unknown)
            agent_types = torch.full((history.shape[0],), self.agent_type_mapping['unknown'], 
                                     dtype=torch.long)
        
        if self.split == 'train':
            future = torch.tensor(self.future_normalized[idx], dtype=torch.float32)
            return {'history': history, 'future': future, 'agent_types': agent_types}
        else:
            return {'history': history, 'agent_types': agent_types}
        
    def denormalize_positions(self, normalized_positions):
        """
        Convert normalized position values back to the original scale
        
        Args:
            normalized_positions: Tensor or numpy array of shape [..., 2]
                containing normalized x,y coordinates
        
        Returns:
            Denormalized positions in the original coordinate space
        """
        if isinstance(normalized_positions, torch.Tensor):
            return normalized_positions * torch.tensor(self.pos_std, 
                        device=normalized_positions.device) + torch.tensor(self.pos_mean, 
                        device=normalized_positions.device)
        else:
            return normalized_positions * self.pos_std + self.pos_mean
            
    def get_num_agent_types(self):
        """Return the number of agent types"""
        return len(self.agent_type_mapping)