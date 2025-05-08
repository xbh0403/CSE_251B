import torch
import torch.nn as nn

class TrajectoryDecoder(nn.Module):
    """Decoder for generating future trajectory predictions"""
    
    def __init__(self, input_dim, hidden_dim, output_seq_len=60, output_dim=2):
        super(TrajectoryDecoder, self).__init__()
        
        self.output_seq_len = output_seq_len
        
        # MLP for decoding trajectory
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_seq_len * output_dim)
        )
    
    def forward(self, x):
        # x shape: [batch_size, input_dim]
        
        # Apply MLP
        batch_size = x.size(0)
        output = self.mlp(x)
        
        # Reshape to [batch_size, output_seq_len, output_dim]
        output = output.view(batch_size, self.output_seq_len, -1)
        
        return output