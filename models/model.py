import torch
import torch.nn as nn

class LSTMEncoder(nn.Module):
    """Simple LSTM for encoding temporal patterns"""
    
    def __init__(self, input_dim=6, hidden_dim=128, output_dim=60*2):
        super(LSTMEncoder, self).__init__()
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        
        # Output projection
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        
        # Apply LSTM
        lstm_out, _ = self.lstm(x)
        
        # Use the final timestep for prediction
        final_hidden = lstm_out[:, -1, :]
        
        # Project to output dimension
        output = self.fc(final_hidden)
        
        # Reshape to trajectory format [batch_size, output_seq_len, 2]
        output = output.view(-1, 60, 2)
        
        return output