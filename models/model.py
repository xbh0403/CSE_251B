import torch
import torch.nn as nn
from .sequence_modules import LSTMEncoder, GRUEncoder, TransformerEncoder

class LSTMModel(nn.Module):
    """LSTM model for trajectory prediction"""
    
    def __init__(self, input_dim=6, hidden_dim=128, output_seq_len=60, output_dim=2):
        super(LSTMModel, self).__init__()
        
        # LSTM Encoder
        self.lstm_encoder = LSTMEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_seq_len * output_dim
        )
    
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
        batch_size, num_agents, seq_len, feat_dim = history.shape
        
        # Extract ego agent only
        ego_history = history[:, 0, :, :]
        
        # Apply LSTM encoder
        predictions = self.lstm_encoder(ego_history)
        
        return predictions
    

class GRUModel(nn.Module):
    """GRU model for trajectory prediction"""
    
    def __init__(self, input_dim=6, hidden_dim=128, output_seq_len=60, output_dim=2):
        super(GRUModel, self).__init__()
        
        # GRU Encoder
        self.gru_encoder = GRUEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_seq_len * output_dim
        )
    
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
        batch_size, num_agents, seq_len, feat_dim = history.shape
        
        # Extract ego agent only
        ego_history = history[:, 0, :, :]
        
        # Apply GRU encoder
        predictions = self.gru_encoder(ego_history)
        
        return predictions


class TransformerModel(nn.Module):
    """Transformer model for trajectory prediction"""
    
    def __init__(self, input_dim=6, hidden_dim=128, output_seq_len=60, output_dim=2, num_heads=4, num_layers=2, dropout=0.1):
        super(TransformerModel, self).__init__()
        
        # Transformer Encoder
        self.transformer_encoder = TransformerEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_seq_len * output_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )
    
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
        batch_size, num_agents, seq_len, feat_dim = history.shape
        
        # Extract ego agent only
        ego_history = history[:, 0, :, :]
        
        # Apply Transformer encoder
        predictions = self.transformer_encoder(ego_history)
        
        return predictions
