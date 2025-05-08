import torch
import torch.nn as nn
from data_utils.feature_engineering import process_batch_radius, compute_constant_velocity, compute_physics_features
from .gnn_modules import GNNEncoder
from .sequence_modules import TransformerEncoder, LSTMEncoder
from .trajectory_decoder import TrajectoryDecoder

class GNNSequenceModel(nn.Module):
    """Combined GNN and Sequence model for trajectory prediction with physics integration"""
    
    def __init__(self, 
                 node_dim=6,           # Original feature dimension
                 gnn_hidden_dim=64,    # GNN hidden dimension
                 seq_hidden_dim=128,   # Sequence model hidden dimension
                 output_seq_len=60,    # Number of future timesteps to predict
                 output_dim=2,         # Output dimension (x, y coordinates)
                 use_transformer=True, # Whether to use Transformer or LSTM
                 num_agent_types=10,   # Number of agent types
                 agent_embedding_dim=16, # Dimension for agent type embeddings
                 physics_dim=6,        # Dimension for physics features
                 use_physics=True      # Whether to use physics-based features
                ):
        super(GNNSequenceModel, self).__init__()
        
        # Agent type embedding
        self.agent_embedding = nn.Embedding(num_agent_types, agent_embedding_dim)
        
        # Whether to use physics-based features
        self.use_physics = use_physics
        
        # Adjust node dimension to include agent type embedding and physics features
        node_dim_with_type = node_dim + agent_embedding_dim
        if use_physics:
            node_dim_with_type += physics_dim
        
        # GNN Encoder
        self.gnn_encoder = GNNEncoder(node_dim_with_type, gnn_hidden_dim)
        
        # Sequence Model
        if use_transformer:
            self.seq_encoder = TransformerEncoder(gnn_hidden_dim, d_model=seq_hidden_dim)
        else:
            self.seq_encoder = LSTMEncoder(gnn_hidden_dim, seq_hidden_dim)
        
        # Trajectory Decoder
        self.trajectory_decoder = TrajectoryDecoder(
            seq_hidden_dim + (output_seq_len * output_dim if use_physics else 0),  # Add constant velocity predictions
            seq_hidden_dim,
            output_seq_len,
            output_dim
        )
        
        # Physics integration layers
        if use_physics:
            self.physics_projection = nn.Linear(output_seq_len * output_dim, seq_hidden_dim)
            self.physics_weight = nn.Parameter(torch.tensor([0.3]), requires_grad=True)
    
    def forward(self, data):
        """
        Forward pass through the network
        
        Args:
            data: Dictionary containing:
                - history: Shape [batch_size, num_agents, seq_len, feature_dim]
                - agent_types: Shape [batch_size, num_agents]
                
        Returns:
            Predicted trajectories: Shape [batch_size, output_seq_len, output_dim]
        """
        batch_size, num_agents, seq_len, feat_dim = data['history'].shape
        device = data['history'].device
        
        # Process batch to get GNN inputs
        node_feats, edge_index, edge_attr = process_batch_radius(data['history'])
        
        # Get agent type embeddings
        if 'agent_types' in data:
            agent_types = data['agent_types'].reshape(-1)  # Flatten to [batch_size * num_agents]
            agent_embeddings = self.agent_embedding(agent_types)  # [batch_size * num_agents, embedding_dim]
        else:
            # Use default embedding if agent types not provided
            agent_types = torch.zeros(batch_size * num_agents, dtype=torch.long, device=device)
            agent_embeddings = self.agent_embedding(agent_types)
        
        # Expand agent embeddings to match sequence length
        agent_embeddings = agent_embeddings.unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size * num_agents, seq_len, embedding_dim]
        
        # Combine with node features
        node_feats_enhanced = torch.cat([node_feats, agent_embeddings], dim=2)
        
        # Add physics-based features if enabled
        if self.use_physics:
            # Compute physics features
            physics_features = compute_physics_features(data['history'])
            physics_features_flat = physics_features.reshape(batch_size * num_agents, -1)
            
            # Expand to match sequence length
            physics_features_expanded = physics_features_flat.unsqueeze(1).expand(-1, seq_len, -1)
            
            # Combine with node features
            node_feats_enhanced = torch.cat([node_feats_enhanced, physics_features_expanded], dim=2)
        
        # Apply GNN for spatial/social reasoning
        gnn_encoded = self.gnn_encoder(node_feats_enhanced.reshape(-1, node_feats_enhanced.size(2)), 
                                       edge_index, edge_attr)
        
        # Reshape back to [batch_size * num_agents, seq_len, gnn_hidden_dim]
        gnn_encoded = gnn_encoded.view(batch_size * num_agents, seq_len, -1)
        
        # Apply sequence model for temporal reasoning
        seq_encoded = self.seq_encoder(gnn_encoded)
        
        # Extract features for the focal agent (index 0 in each scene)
        focal_indices = torch.arange(0, batch_size * num_agents, num_agents, device=device)
        focal_features = seq_encoded[focal_indices]
        
        # For trajectory prediction, we use the final timestep features
        final_features = focal_features[:, -1, :]
        
        # Compute constant velocity baseline for each focal agent
        if self.use_physics:
            const_vel_pred = compute_constant_velocity(data['history'])
            const_vel_flat = const_vel_pred.reshape(batch_size, -1)
            
            # Project constant velocity predictions and combine with final features
            cv_projected = self.physics_projection(const_vel_flat)
            combined_features = torch.cat([final_features, const_vel_flat], dim=1)
        else:
            combined_features = final_features
        
        # Decode to get future trajectory
        neural_trajectory = self.trajectory_decoder(combined_features)
        
        if self.use_physics:
            # Combine neural trajectory with constant velocity prediction
            # The weight is a learnable parameter
            weight = torch.sigmoid(self.physics_weight)  # Ensure weight is between 0 and 1
            final_trajectory = weight * neural_trajectory + (1 - weight) * const_vel_pred
            return final_trajectory
        else:
            return neural_trajectory