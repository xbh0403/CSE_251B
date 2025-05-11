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
            num_layers=2,
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
    
class GRUEncoder(nn.Module):
    """Simple GRU for encoding temporal patterns"""
    
    def __init__(self, input_dim=6, hidden_dim=128, output_dim=60*2):
        super(GRUEncoder, self).__init__()
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )
        
        # Output projection
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        
        # Apply GRU
        gru_out, _ = self.gru(x)
        
        # Use the final timestep for prediction
        final_hidden = gru_out[:, -1, :]
        
        # Project to output dimension
        output = self.fc(final_hidden)
        
        # Reshape to trajectory format [batch_size, output_seq_len, 2]
        output = output.view(-1, 60, 2)
        
        return output
    

class TransformerEncoder(nn.Module):
    """Transformer for encoding temporal patterns"""
    
    def __init__(self, input_dim=6, hidden_dim=128, output_dim=60*2, num_heads=4, num_layers=2, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.zeros(1, 50, hidden_dim))  # 50 is the sequence length
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )
        
        # Output projection
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        
        # Project input to hidden dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x + self.pos_encoder[:, :x.size(1), :]
        
        # Apply transformer encoder
        transformer_out = self.transformer_encoder(x)
        
        # Use the final timestep for prediction
        final_hidden = transformer_out[:, -1, :]
        
        # Project to output dimension
        output = self.fc(final_hidden)
        
        # Reshape to trajectory format [batch_size, output_seq_len, 2]
        output = output.view(-1, 60, 2)
        
        return output

