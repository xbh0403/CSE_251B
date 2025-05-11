import torch
import torch.nn as nn

class PhysicsGuidedLSTM(nn.Module):
    """
    A simplified physics-guided LSTM model for trajectory prediction.
    This version is designed to be a drop-in replacement for the existing LSTM model.
    """
    
    def __init__(self, input_dim=6, hidden_dim=128, output_seq_len=60, output_dim=2):
        super(PhysicsGuidedLSTM, self).__init__()
        
        # Main LSTM encoder
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            num_layers=2,
            dropout=0.1
        )
        
        # Separate prediction branches for different aspects of motion
        self.position_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_seq_len * output_dim)
        )
        
        self.output_seq_len = output_seq_len
        self.output_dim = output_dim
    
    def forward(self, data):
        """
        Forward pass through the network
        
        Args:
            data: Dictionary containing:
                - history: Shape [batch_size, num_agents, seq_len, feature_dim]
                
        Returns:
            Predicted trajectories: Shape [batch_size, output_seq_len, output_dim]
        """
        # Extract history data
        history = data['history']
        
        # Handle different input formats (with or without num_agents dimension)
        if history.dim() == 4:
            batch_size, num_agents, seq_len, feat_dim = history.shape
            # Extract ego agent only
            ego_history = history[:, 0, :, :]
        else:
            batch_size, seq_len, feat_dim = history.shape
            ego_history = history
        
        # Process through LSTM
        lstm_out, _ = self.lstm(ego_history)
        
        # Use the final hidden state for prediction
        final_hidden = lstm_out[:, -1, :]
        
        # Generate the position predictions
        position_output = self.position_predictor(final_hidden)
        position_output = position_output.view(batch_size, self.output_seq_len, self.output_dim)
        
        return position_output