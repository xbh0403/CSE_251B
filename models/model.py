import torch
import torch.nn as nn
import torch.nn.functional as F
from .sequence_modules import (
    LSTMEncoder, LSTMDecoder, GRUEncoder, GRUDecoder, 
    TransformerEncoder, TransformerDecoder
)
from .social_modules import SocialContextEncoder

class Seq2SeqLSTMModel(nn.Module):
    """Sequence-to-sequence LSTM model for trajectory prediction"""
    
    def __init__(self, input_dim=6, hidden_dim=128, output_seq_len=60, output_dim=2, num_layers=2):
        super(Seq2SeqLSTMModel, self).__init__()
        
        # Encoder and decoder
        self.encoder = LSTMEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )
        
        self.decoder = LSTMDecoder(
            input_dim=output_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers
        )
        
        # Parameters
        self.output_seq_len = output_seq_len
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
    
    def forward(self, data, teacher_forcing_ratio=0.5):
        """
        Forward pass through the network
        
        Args:
            data: Dictionary containing:
                - history: Shape [batch_size, num_agents, seq_len, feature_dim]
                - future: Shape [batch_size, output_seq_len, output_dim] (only in training)
            teacher_forcing_ratio: Probability of using teacher forcing (0-1)
                
        Returns:
            Predicted trajectories: Shape [batch_size, output_seq_len, output_dim]
        """
        # Extract history data
        history = data['history']
        batch_size = history.shape[0]
        
        # Extract ego agent only
        ego_history = history[:, 0, :, :]
        
        # Encode the history sequence
        _, hidden = self.encoder(ego_history)
        
        # Determine if we're in training or inference mode
        use_teacher_forcing = self.training and 'future' in data and torch.rand(1).item() < teacher_forcing_ratio

        # Initialize output container
        predictions = torch.zeros(batch_size, self.output_seq_len, self.output_dim, 
                                 device=history.device)
        
        # FIXED: Initialize decoder input with the last position from history (x,y coordinates)
        decoder_input = ego_history[:, -1:, :self.output_dim]
        
        # Iteratively decode the sequence
        for t in range(self.output_seq_len):
            # Get prediction for current timestep
            output, hidden = self.decoder(decoder_input, hidden)
            
            # Store prediction
            predictions[:, t:t+1, :] = output
            
            # Update decoder input for next timestep
            if use_teacher_forcing and t < self.output_seq_len - 1:
                # Use ground truth as next input
                decoder_input = data['future'][:, t:t+1, :]
            else:
                # Use current prediction as next input
                decoder_input = output
        
        return predictions


class Seq2SeqGRUModel(nn.Module):
    """Sequence-to-sequence GRU model for trajectory prediction"""
    
    def __init__(self, input_dim=6, hidden_dim=128, output_seq_len=60, output_dim=2, num_layers=2):
        super(Seq2SeqGRUModel, self).__init__()
        
        # Encoder and decoder
        self.encoder = GRUEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )
        
        self.decoder = GRUDecoder(
            input_dim=output_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers
        )
        
        # Parameters
        self.output_seq_len = output_seq_len
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
    
    def forward(self, data, teacher_forcing_ratio=0.5):
        """
        Forward pass through the network
        
        Args:
            data: Dictionary containing:
                - history: Shape [batch_size, num_agents, seq_len, feature_dim]
                - future: Shape [batch_size, output_seq_len, output_dim] (only in training)
            teacher_forcing_ratio: Probability of using teacher forcing (0-1)
                
        Returns:
            Predicted trajectories: Shape [batch_size, output_seq_len, output_dim]
        """
        # Extract history data
        history = data['history']
        batch_size = history.shape[0]
        
        # Extract ego agent only
        ego_history = history[:, 0, :, :]
        
        # Encode the history sequence
        _, hidden = self.encoder(ego_history)
        
        # Determine if we're in training or inference mode
        use_teacher_forcing = self.training and 'future' in data and torch.rand(1).item() < teacher_forcing_ratio
        
        # Initialize output container
        predictions = torch.zeros(batch_size, self.output_seq_len, self.output_dim, 
                                 device=history.device)
        
        # FIXED: Initialize decoder input with the last position from history (x,y coordinates)
        decoder_input = ego_history[:, -1:, :self.output_dim]
        
        # Iteratively decode the sequence
        for t in range(self.output_seq_len):
            # Get prediction for current timestep
            output, hidden = self.decoder(decoder_input, hidden)
            
            # Store prediction
            predictions[:, t:t+1, :] = output
            
            # Update decoder input for next timestep
            if use_teacher_forcing and t < self.output_seq_len - 1:
                # Use ground truth as next input
                decoder_input = data['future'][:, t:t+1, :]
            else:
                # Use current prediction as next input
                decoder_input = output
        
        return predictions


class Seq2SeqTransformerModel(nn.Module):
    """Sequence-to-sequence Transformer model for trajectory prediction"""
    
    def __init__(self, input_dim=6, hidden_dim=128, output_seq_len=60, output_dim=2, 
                 num_heads=4, num_layers=2, dropout=0.1):
        super(Seq2SeqTransformerModel, self).__init__()
        
        # Encoder and decoder
        self.encoder = TransformerEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )
        
        self.decoder = TransformerDecoder(
            input_dim=output_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Parameters
        self.output_seq_len = output_seq_len
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        # Initial input projection for decoder inputs
        self.input_projection = nn.Linear(output_dim, hidden_dim)
    
    def _generate_square_subsequent_mask(self, sz):
        """Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self, data, teacher_forcing_ratio=0.5):
        """
        Forward pass through the network
        
        Args:
            data: Dictionary containing:
                - history: Shape [batch_size, num_agents, seq_len, feature_dim]
                - future: Shape [batch_size, output_seq_len, output_dim] (only in training)
            teacher_forcing_ratio: Probability of using teacher forcing (0-1)
                
        Returns:
            Predicted trajectories: Shape [batch_size, output_seq_len, output_dim]
        """
        # Extract history data
        history = data['history']
        batch_size = history.shape[0]
        device = history.device
        
        # Extract ego agent only
        ego_history = history[:, 0, :, :]
        
        # Encode the history sequence
        memory = self.encoder(ego_history)
        
        # Determine if we're in training or inference mode
        use_teacher_forcing = self.training and 'future' in data and torch.rand(1).item() < teacher_forcing_ratio
        
        # FIXED: Extract the last position from history (x,y coordinates)
        last_position = ego_history[:, -1:, :self.output_dim]
        
        if use_teacher_forcing:
            # Teacher forcing - use all ground truth as input
            # But prepend last position from history to the sequence
            
            if 'future' in data:
                # Use only the first n-1 future points (we predict the next one)
                # and prepend the last observed position
                tgt_input = torch.cat([last_position, data['future'][:, :-1, :]], dim=1)
            else:
                # Fallback - only use last position (first timestep)
                tgt_input = last_position
                
            # Create causal mask for transformer decoder
            tgt_mask = self._generate_square_subsequent_mask(tgt_input.size(1)).to(device)
            
            # Get predictions all at once
            predictions = self.decoder(tgt_input, memory, tgt_mask=tgt_mask)
            
        else:
            # Autoregressive generation - one timestep at a time
            predictions = torch.zeros(batch_size, self.output_seq_len, self.output_dim, device=device)
            
            # Start with just the last observed position
            tgt_input = last_position
            
            for t in range(self.output_seq_len):
                # Create causal mask for transformer decoder
                tgt_mask = self._generate_square_subsequent_mask(tgt_input.size(1)).to(device)
                
                # Get predictions for all timesteps so far
                output = self.decoder(tgt_input, memory, tgt_mask=tgt_mask)
                
                # Store the prediction for the current timestep
                predictions[:, t:t+1, :] = output[:, -1:, :]
                
                # Update decoder input for next timestep
                if t < self.output_seq_len - 1:
                    # Append current prediction to the sequence
                    tgt_input = torch.cat([tgt_input, output[:, -1:, :]], dim=1)
        
        return predictions
    

class SocialGRUModel(nn.Module):
    """
    GRU-based model that incorporates social context for trajectory prediction
    """
    
    def __init__(self, input_dim=6, hidden_dim=128, output_seq_len=60, output_dim=2, num_layers=2):
        super(SocialGRUModel, self).__init__()
        
        # Social context encoder
        self.social_encoder = SocialContextEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=4
        )
        
        # Temporal encoder for processed trajectories
        self.encoder = GRUEncoder(
            input_dim=hidden_dim,  # Takes output from social encoder
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )
        
        # Trajectory decoder
        self.decoder = GRUDecoder(
            input_dim=output_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers
        )
        
        # Parameters
        self.output_seq_len = output_seq_len
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
    
    def forward(self, data, teacher_forcing_ratio=0.5):
        """
        Forward pass through the network
        
        Args:
            data: Dictionary containing:
                - history: Shape [batch_size, num_agents, seq_len, feature_dim]
                - future: Shape [batch_size, output_seq_len, output_dim] (only in training)
            teacher_forcing_ratio: Probability of using teacher forcing (0-1)
                
        Returns:
            Predicted trajectories: Shape [batch_size, output_seq_len, output_dim]
        """
        # Extract history data
        history = data['history']
        batch_size = history.shape[0]
        device = history.device
        
        # Process with social context encoder
        social_features = self.social_encoder(history)  # [batch_size, seq_len, hidden_dim]
        
        # Encode the trajectory with social context
        _, hidden = self.encoder(social_features)
        
        # Determine if we're in training or inference mode
        use_teacher_forcing = self.training and 'future' in data and torch.rand(1).item() < teacher_forcing_ratio
        
        # Initialize output container
        predictions = torch.zeros(batch_size, self.output_seq_len, self.output_dim, 
                                 device=device)
        
        # Initialize decoder input with the last position from history (x,y coordinates)
        decoder_input = history[:, 0, -1:, :self.output_dim]
        
        # Iteratively decode the sequence
        for t in range(self.output_seq_len):
            # Get prediction for current timestep
            output, hidden = self.decoder(decoder_input, hidden)
            
            # Store prediction
            predictions[:, t:t+1, :] = output
            
            # Update decoder input for next timestep
            if use_teacher_forcing and t < self.output_seq_len - 1:
                # Use ground truth as next input
                decoder_input = data['future'][:, t:t+1, :]
            else:
                # Use current prediction as next input
                decoder_input = output
        
        return predictions