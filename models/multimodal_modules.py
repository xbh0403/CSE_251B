import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiModalGRUDecoder(nn.Module):
    """
    GRU decoder that generates multiple possible future trajectories
    with confidence scores for each
    """
    
    def __init__(self, input_dim=2, hidden_dim=128, output_dim=2, num_layers=2, num_modes=3):
        super(MultiModalGRUDecoder, self).__init__()
        
        # Core GRU decoder (shared across modes)
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Mode-specific projection layers
        self.mode_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            ) for _ in range(num_modes)
        ])
        
        # Confidence scoring for each mode
        self.confidence_scorer = nn.Sequential(
            nn.Linear(hidden_dim * num_layers, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_modes),
            # No softmax here - will apply in forward pass
        )
        
        self.num_modes = num_modes
        self.output_dim = output_dim
    
    def forward(self, x, hidden):
        """
        Forward pass generating multiple trajectory hypotheses
        
        Args:
            x: Input tensor [batch_size, 1, input_dim] or [batch_size, input_dim]
            hidden: Hidden state from encoder
            
        Returns:
            predictions: Multiple predicted trajectories [batch_size, 1, num_modes, output_dim]
            confidences: Confidence score for each mode [batch_size, num_modes]
        """
        # Ensure input has the right shape for GRU [batch_size, seq_len, input_dim]
        if x.dim() == 4:  # [batch_size, 1, 1, input_dim]
            x = x.squeeze(2)  # -> [batch_size, 1, input_dim]
        elif x.dim() == 3 and x.size(1) == 1 and x.size(2) == 2:
            # This is correct already: [batch_size, 1, input_dim]
            pass
        elif x.dim() == 2:  # [batch_size, input_dim]
            x = x.unsqueeze(1)  # -> [batch_size, 1, input_dim]
        
        # Apply GRU
        output, hidden_state = self.gru(x, hidden)  # output: [batch_size, 1, hidden_dim]
        
        # Generate confidence scores from the final hidden state
        # Reshape hidden state: [num_layers, batch_size, hidden_dim] -> [batch_size, num_layers*hidden_dim]
        hidden_for_confidence = hidden_state.transpose(0, 1).contiguous().view(hidden_state.size(1), -1)
        
        # Generate raw confidence scores
        confidences = self.confidence_scorer(hidden_for_confidence)  # [batch_size, num_modes]
        confidence_probs = F.softmax(confidences, dim=1)  # Apply softmax to get probabilities
        
        # Generate predictions for each mode
        mode_predictions = []
        for i in range(self.num_modes):
            # Apply mode-specific projection
            mode_pred = self.mode_projections[i](output)  # [batch_size, 1, output_dim]
            mode_predictions.append(mode_pred)
        
        # Stack all mode predictions
        # shape: [batch_size, 1, num_modes, output_dim]
        stacked_predictions = torch.stack(mode_predictions, dim=2)
        
        return stacked_predictions, confidence_probs, hidden_state