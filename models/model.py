import torch
import torch.nn as nn
import torch.nn.functional as F
from .sequence_modules import (
    LSTMEncoder, LSTMDecoder, GRUEncoder, GRUDecoder, 
    TransformerEncoder, TransformerDecoder
)
from .social_modules import SocialContextEncoder
from .multimodal_modules import MultiModalGRUDecoder

class Seq2SeqLSTMModel(nn.Module):
    """Sequence-to-sequence LSTM model for trajectory prediction, with a learnable start token."""

    def __init__(self, input_dim=6, hidden_dim=128, output_seq_len=60, output_dim=2, num_layers=2):
        super(Seq2SeqLSTMModel, self).__init__()

        # Encoder 和 Decoder
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

        # 可学习的 start token，初始为 (1, 1, output_dim)，训练时会被更新
        self.start_token = nn.Parameter(torch.randn(1, 1, output_dim) * 0.1)

        # 其它参数
        self.output_seq_len = output_seq_len
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

    def forward(self, data, teacher_forcing_ratio=0.5):
        """
        Forward pass through the network

        Args:
            data: Dictionary containing:
                - history: Shape [batch_size, num_agents, seq_len, feature_dim]
                - future: Shape [batch_size, output_seq_len, output_dim] (only 在训练时使用)
            teacher_forcing_ratio: 使用 teacher forcing 的概率 (0-1)

        Returns:
            predictions: Tensor of shape [batch_size, output_seq_len, output_dim]
        """
        history = data['history']                          # [B, num_agents, seq_len, feature_dim]
        batch_size = history.shape[0]

        # 只取 Ego Agent 的历史 (B, seq_len, feature_dim)
        ego_history = history[:, 0, :, :]

        # Encoder：得到最后一个隐藏状态 hidden
        _, hidden = self.encoder(ego_history)

        # 判断是否使用 teacher forcing
        use_teacher_forcing = (
            self.training
            and 'future' in data
            and torch.rand(1).item() < teacher_forcing_ratio
        )

        # 用于保存输出
        predictions = torch.zeros(
            batch_size,
            self.output_seq_len,
            self.output_dim,
            device=history.device
        )

        # ——— 关键改动 ——–
        # 用可学习的 start_token 作为 Decoder 的第一个输入
        # self.start_token 的 shape=(1,1,output_dim)，.expand 后变为 (batch_size,1,output_dim)
        decoder_input = self.start_token.expand(batch_size, -1, -1)
        # ————————————

        for t in range(self.output_seq_len):
            # Decoder 前向：输入 decoder_input + 上一个 hidden
            output, hidden = self.decoder(decoder_input, hidden)

            # 存储当前时刻的预测
            # output shape = [batch_size, 1, output_dim]
            predictions[:, t:t+1, :] = output

            # 更新下一步的 decoder_input
            if use_teacher_forcing and t < self.output_seq_len - 1:
                # 如果用 teacher forcing，下一步输入是真实 ground-truth
                decoder_input = data['future'][:, t:t+1, :]
            else:
                # 否则用模型当前预测的结果
                decoder_input = output

        return predictions
# class Seq2SeqLSTMModel(nn.Module):
#     """Sequence-to-sequence LSTM model for trajectory prediction"""
    
#     def __init__(self, input_dim=6, hidden_dim=128, output_seq_len=60, output_dim=2, num_layers=2):
#         super(Seq2SeqLSTMModel, self).__init__()
        
#         # Encoder and decoder
#         self.encoder = LSTMEncoder(
#             input_dim=input_dim,
#             hidden_dim=hidden_dim,
#             num_layers=num_layers
#         )
        
#         self.decoder = LSTMDecoder(
#             input_dim=output_dim,
#             hidden_dim=hidden_dim,
#             output_dim=output_dim,
#             num_layers=num_layers
#         )
        
#         # Parameters
#         self.output_seq_len = output_seq_len
#         self.output_dim = output_dim
#         self.hidden_dim = hidden_dim
    
#     def forward(self, data, teacher_forcing_ratio=0.5):
#         """
#         Forward pass through the network
        
#         Args:
#             data: Dictionary containing:
#                 - history: Shape [batch_size, num_agents, seq_len, feature_dim]
#                 - future: Shape [batch_size, output_seq_len, output_dim] (only in training)
#             teacher_forcing_ratio: Probability of using teacher forcing (0-1)
                
#         Returns:
#             Predicted trajectories: Shape [batch_size, output_seq_len, output_dim]
#         """
#         # Extract history data
#         history = data['history']
#         batch_size = history.shape[0]
        
#         # Extract ego agent only
#         ego_history = history[:, 0, :, :]
        
#         # Encode the history sequence
#         _, hidden = self.encoder(ego_history)
        
#         # Determine if we're in training or inference mode
#         use_teacher_forcing = self.training and 'future' in data and torch.rand(1).item() < teacher_forcing_ratio

#         # Initialize output container
#         predictions = torch.zeros(batch_size, self.output_seq_len, self.output_dim, 
#                                  device=history.device)
        
#         # FIXED: Initialize decoder input with the last position from history (x,y coordinates)
#         decoder_input = ego_history[:, -1:, :self.output_dim]
        
#         # Iteratively decode the sequence
#         for t in range(self.output_seq_len):
#             # Get prediction for current timestep
#             output, hidden = self.decoder(decoder_input, hidden)
            
#             # Store prediction
#             predictions[:, t:t+1, :] = output
            
#             # Update decoder input for next timestep
#             if use_teacher_forcing and t < self.output_seq_len - 1:
#                 # Use ground truth as next input
#                 decoder_input = data['future'][:, t:t+1, :]
#             else:
#                 # Use current prediction as next input
#                 decoder_input = output
        
#         return predictions


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
    

class MultiModalGRUModel(nn.Module):
    """
    GRU-based model that predicts multiple possible future trajectories
    """
    
    def __init__(self, input_dim=6, hidden_dim=128, output_seq_len=60, output_dim=2, 
                 num_layers=2, num_modes=3):
        super(MultiModalGRUModel, self).__init__()
        
        # Can optionally use the social encoder
        self.use_social = True
        
        if self.use_social:
            # Social context encoder
            self.social_encoder = SocialContextEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_heads=4
            )
            
            # Temporal encoder processes social features
            self.encoder = GRUEncoder(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers
            )
        else:
            # Temporal encoder for ego agent only
            self.encoder = GRUEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers
            )
        
        # Multi-modal decoder
        self.decoder = MultiModalGRUDecoder(
            input_dim=output_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            num_modes=num_modes
        )
        
        # Parameters
        self.output_seq_len = output_seq_len
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_modes = num_modes
    
    def forward(self, data, teacher_forcing_ratio=0.5, mode_idx=None):
        """
        Forward pass through the network
        
        Args:
            data: Dictionary containing:
                - history: Shape [batch_size, num_agents, seq_len, feature_dim]
                - future: Shape [batch_size, output_seq_len, output_dim] (only in training)
            teacher_forcing_ratio: Probability of using teacher forcing (0-1)
            mode_idx: If provided, returns predictions only for this specific mode
                
        Returns:
            During training: 
                predictions: Shape [batch_size, output_seq_len, num_modes, output_dim]
                confidences: Shape [batch_size, num_modes]
            During inference with mode_idx:
                predictions: Shape [batch_size, output_seq_len, output_dim]
        """
        # Extract history data
        history = data['history']
        batch_size = history.shape[0]
        device = history.device
        
        # Process with encoder
        if self.use_social:
            # Process with social context encoder
            social_features = self.social_encoder(history)  # [batch_size, seq_len, hidden_dim]
            # Encode the trajectory with social context
            _, hidden = self.encoder(social_features)
        else:
            # Extract ego agent only
            ego_history = history[:, 0, :, :]
            # Encode the history sequence
            _, hidden = self.encoder(ego_history)
        
        # Determine if we're in training or inference mode
        use_teacher_forcing = self.training and 'future' in data and torch.rand(1).item() < teacher_forcing_ratio
        
        # Initialize output containers
        predictions = torch.zeros(batch_size, self.output_seq_len, self.num_modes, self.output_dim, 
                                 device=device)
        all_confidences = []
        
        # Initialize decoder input with the last position from history (x,y coordinates)
        # FIXED: Proper tensor shape handling
        decoder_input = history[:, 0, -1, :self.output_dim].unsqueeze(1)  # [batch_size, 1, output_dim]
        
        # Iteratively decode the sequence
        for t in range(self.output_seq_len):
            # Get multi-modal predictions for current timestep
            output, confidences, hidden = self.decoder(decoder_input, hidden)
            
            # Store predictions and confidences
            predictions[:, t:t+1, :, :] = output
            if t == 0:  # Store confidence scores (same for all timesteps)
                all_confidences = confidences
            
            # Update decoder input for next timestep
            if use_teacher_forcing and t < self.output_seq_len - 1:
                # Use ground truth as next input
                decoder_input = data['future'][:, t:t+1, :]
            else:
                # Use highest confidence prediction as next input
                if mode_idx is not None:
                    # Use the specified mode
                    decoder_input = output[:, :, mode_idx, :]
                else:
                    # Use highest confidence mode
                    best_mode = torch.argmax(confidences, dim=1)
                    decoder_input = torch.gather(
                        output.squeeze(1),  # [batch_size, num_modes, output_dim]
                        dim=1,
                        index=best_mode.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, self.output_dim)
                    ).unsqueeze(1)  # Add back sequence dimension
        
        # If specific mode requested for inference, return just that mode
        if mode_idx is not None:
            return predictions[:, :, mode_idx, :]
        
        # For training, return all modes and confidences
        return predictions, all_confidences