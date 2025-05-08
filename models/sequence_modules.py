import torch
import torch.nn as nn
import math

class LSTMEncoder(nn.Module):
    """LSTM for encoding temporal patterns"""
    
    def __init__(self, input_dim, hidden_dim, num_layers=2, bidirectional=True):
        super(LSTMEncoder, self).__init__()
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # Output dimension adjustment if bidirectional
        self.output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(self.output_dim, hidden_dim)
    
    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        
        # Apply LSTM
        outputs, (hidden, cell) = self.lstm(x)
        
        # Use the output of each timestep
        outputs = self.fc(outputs)
        
        return outputs
    

class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer"""
    
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register buffer (not a parameter, but part of the module)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        return x + self.pe[:, :x.size(1)]

class TransformerEncoder(nn.Module):
    """Transformer for encoding temporal patterns"""
    
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=4, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        
        # Project input to d_model dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer
        x = self.transformer(x)
        
        # Final projection
        x = self.output_projection(x)
        
        return x    
