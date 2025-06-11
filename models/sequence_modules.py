import torch
import torch.nn as nn

class LSTMEncoder(nn.Module):
    """LSTM for encoding temporal patterns"""
    
    def __init__(self, input_dim=6, hidden_dim=128, num_layers=2):
        super(LSTMEncoder, self).__init__()
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
    
    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        
        # Apply LSTM and return all hidden states and cell states
        output, (hidden, cell) = self.lstm(x)
        
        # Return both the full sequence of outputs and the final states
        return output, (hidden, cell)


class LSTMDecoder(nn.Module):
    """LSTM for decoding and generating trajectory predictions"""
    
    def __init__(self, input_dim=2, hidden_dim=128, output_dim=2, num_layers=2):
        super(LSTMDecoder, self).__init__()
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Output projection
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, hidden):
        # x shape: [batch_size, 1, input_dim] - just one timestep
        # hidden is the tuple (hidden_state, cell_state) from encoder or previous step
        
        # Apply LSTM
        output, hidden_state = self.lstm(x, hidden)
        
        # Project to output dimension
        prediction = self.fc(output)
        
        return prediction, hidden_state


class GRUEncoder(nn.Module):
    """GRU for encoding temporal patterns"""
    
    def __init__(self, input_dim=6, hidden_dim=128, num_layers=2):
        super(GRUEncoder, self).__init__()
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
    
    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        
        # Apply GRU
        output, hidden = self.gru(x)
        
        # Return both the full sequence of outputs and the final state
        return output, hidden


class GRUDecoder(nn.Module):
    """GRU for decoding and generating trajectory predictions"""
    
    def __init__(self, input_dim=2, hidden_dim=128, output_dim=2, num_layers=2):
        super(GRUDecoder, self).__init__()
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Output projection
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, hidden):
        # x shape: [batch_size, 1, input_dim] - just one timestep
        # hidden is the hidden state from encoder or previous step
        
        # Apply GRU
        output, hidden_state = self.gru(x, hidden)
        
        # Project to output dimension
        prediction = self.fc(output)
        
        return prediction, hidden_state


class TransformerEncoder(nn.Module):
    """Transformer for encoding temporal patterns"""
    
    def __init__(self, input_dim=6, hidden_dim=128, num_heads=4, num_layers=2, dropout=0.1):
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
        
        self.hidden_dim = hidden_dim
    
    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        
        # Project input to hidden dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x + self.pos_encoder[:, :x.size(1), :]
        
        # Apply transformer encoder
        memory = self.transformer_encoder(x)
        
        return memory


class TransformerDecoder(nn.Module):
    """Transformer for decoding and generating trajectory predictions"""
    
    def __init__(self, input_dim=2, hidden_dim=128, output_dim=2, num_heads=4, num_layers=2, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.zeros(1, 60, hidden_dim))  # 60 is max prediction length
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_layers
        )
        
        # Output projection
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, tgt, memory, tgt_mask=None):
        # tgt shape: [batch_size, pred_len, input_dim]
        # memory shape: [batch_size, seq_len, hidden_dim]
        
        # Project target to hidden dimension
        tgt = self.input_projection(tgt)
        
        # Add positional encoding
        tgt = tgt + self.pos_encoder[:, :tgt.size(1), :]
        
        # Apply transformer decoder
        output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)
        
        # Project to output dimension
        prediction = self.fc(output)
        
        return prediction