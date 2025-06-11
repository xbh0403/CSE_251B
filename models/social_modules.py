import torch
import torch.nn as nn

class SocialContextEncoder(nn.Module):
    """
    Encodes social context using attention between the ego vehicle and other agents
    """
    
    def __init__(self, input_dim=6, hidden_dim=128, num_heads=4):
        super(SocialContextEncoder, self).__init__()
        
        # Agent-level feature extraction
        self.agent_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Attention mechanism for agent interaction
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Feature integration projection
        self.feature_fusion = nn.Linear(2*hidden_dim, hidden_dim)
        
    def forward(self, x):
        """
        Process agent histories and extract social context
        
        Args:
            x: Agent trajectories [batch_size, num_agents, seq_len, input_dim]
            
        Returns:
            Enhanced ego features with social context [batch_size, seq_len, hidden_dim]
        """
        batch_size, num_agents, seq_len, _ = x.shape
        
        # Encode each agent's trajectory
        agent_features = []
        for i in range(num_agents):
            # Skip padded agents (all zeros)
            if torch.sum(torch.abs(x[:, i])) > 0:
                # Encode this agent's trajectory
                agent_feat = self.agent_encoder(x[:, i])  # [batch_size, seq_len, hidden_dim]
                agent_features.append(agent_feat)
            else:
                # For padded agents, use zero features
                agent_feat = torch.zeros(batch_size, seq_len, self.agent_encoder[-1].out_features, 
                                        device=x.device)
                agent_features.append(agent_feat)
                
        # Stack agent features
        agent_features = torch.stack(agent_features, dim=1)  # [batch_size, num_agents, seq_len, hidden_dim]
        
        # Get features at final timestep for attention
        last_step_features = agent_features[:, :, -1, :]  # [batch_size, num_agents, hidden_dim]
        
        # Separate ego from other agents
        ego_features = last_step_features[:, 0:1, :]  # [batch_size, 1, hidden_dim]
        other_features = last_step_features[:, 1:, :]  # [batch_size, num_agents-1, hidden_dim]
        
        # Apply attention from ego to other agents
        # If no other valid agents, skip attention
        if torch.sum(torch.abs(other_features)) > 0:
            context, _ = self.attention(
                query=ego_features,
                key=other_features,
                value=other_features
            )
            
            # Combine ego features with social context
            combined = torch.cat([ego_features, context], dim=-1)  # [batch_size, 1, 2*hidden_dim]
            social_context = self.feature_fusion(combined)  # [batch_size, 1, hidden_dim]
        else:
            # If no other agents, just use ego features
            social_context = ego_features
            
        # Get full ego trajectory features
        ego_trajectory = agent_features[:, 0, :, :]  # [batch_size, seq_len, hidden_dim]
        
        # Create social context for each timestep by concatenating with ego features
        # and then projecting back to hidden_dim
        social_context_expanded = social_context.repeat(1, seq_len, 1)  # [batch_size, seq_len, hidden_dim]
        
        # Combine with ego trajectory features for each timestep
        combined_features = torch.cat([ego_trajectory, social_context_expanded], dim=-1)
        enhanced_features = self.feature_fusion(combined_features)  # [batch_size, seq_len, hidden_dim]
        
        return enhanced_features