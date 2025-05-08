import torch
import torch.nn as nn
# from data_utils.feature_engineering_oldslow import process_batch
from data_utils.feature_engineering import process_batch_radius
from .gnn_modules import GNNEncoder
from .sequence_modules import TransformerEncoder, LSTMEncoder
from .trajectory_decoder import TrajectoryDecoder

class GNNSequenceModel(nn.Module):
    """Combined GNN and Sequence model for trajectory prediction"""
    
    def __init__(self, 
                 node_dim=6,           # Original feature dimension
                 gnn_hidden_dim=64,    # GNN hidden dimension
                 seq_hidden_dim=128,   # Sequence model hidden dimension
                 output_seq_len=60,    # Number of future timesteps to predict
                 output_dim=2,         # Output dimension (x, y coordinates)
                 use_transformer=True  # Whether to use Transformer or LSTM
                ):
        super(GNNSequenceModel, self).__init__()
        
        # GNN Encoder
        self.gnn_encoder = GNNEncoder(node_dim, gnn_hidden_dim)
        
        # Sequence Model
        if use_transformer:
            self.seq_encoder = TransformerEncoder(gnn_hidden_dim, d_model=seq_hidden_dim)
        else:
            self.seq_encoder = LSTMEncoder(gnn_hidden_dim, seq_hidden_dim)
        
        # Trajectory Decoder
        self.trajectory_decoder = TrajectoryDecoder(
            seq_hidden_dim, seq_hidden_dim, output_seq_len, output_dim
        )
    
    def forward(self, data):
        """
        Forward pass through the network
        
        Args:
            data: Dictionary containing:
                - history: Shape [batch_size, num_agents, seq_len, feature_dim]
                - (other processed data like edge_index, edge_attr)
                
        Returns:
            Predicted trajectories: Shape [batch_size, output_seq_len, output_dim]
        """
        batch_size, num_agents, seq_len, feat_dim = data['history'].shape
        
        # Process batch to get GNN inputs
        node_feats, edge_index, edge_attr = process_batch_radius(data['history'])
        # print("processed batch done")
        # Apply GNN for spatial/social reasoning
        gnn_encoded = self.gnn_encoder(node_feats.reshape(-1, feat_dim), edge_index, edge_attr)
        # print("gnn encoded done")
        # Reshape back to [batch_size * num_agents, seq_len, gnn_hidden_dim]
        gnn_encoded = gnn_encoded.view(batch_size * num_agents, seq_len, -1)
        # print("reshaped gnn encoded done")
        # Apply sequence model for temporal reasoning
        seq_encoded = self.seq_encoder(gnn_encoded)
        # print("seq encoded done")
        # Extract features for the focal agent (index 0 in each scene)
        # We extract separate indices (batch_size steps apart) to get only the focal agents
        focal_indices = torch.arange(0, batch_size * num_agents, num_agents, device=seq_encoded.device)
        focal_features = seq_encoded[focal_indices]
        # print("focal features done")
        # For trajectory prediction, we use the final timestep features
        final_features = focal_features[:, -1, :]
        # print("final features done")
        # Decode to get future trajectory
        predicted_trajectory = self.trajectory_decoder(final_features)
        # print("predicted trajectory done")
        return predicted_trajectory