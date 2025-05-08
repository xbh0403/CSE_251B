import torch

def process_batch(batch_data, time_steps=50):
    """
    Process batch to extract additional features and create edge connections
    
    Args:
        batch_data: tensor of shape [batch_size, num_agents, time_steps, feature_dim]
        
    Returns:
        node_feats: Node features for GNN
        edge_index: Edge connections (which nodes are connected)
        edge_attr: Edge attributes (features of the connections)
    """
    batch_size, num_agents, seq_len, feat_dim = batch_data.shape
    
    # 1. Create node features - reshape to have each agent at each timestep be a node
    # Original features: [x, y, vx, vy, heading, obj_type]
    node_feats = batch_data.reshape(batch_size * num_agents, seq_len, feat_dim)
    
    # 2. Create edges - connect agents within same scene that are close
    edge_indices = []
    edge_attrs = []
    
    for b in range(batch_size):
        # For each scene in the batch
        for t in range(seq_len):
            # Consider the position at each timestep
            agent_positions = batch_data[b, :, t, :2]  # [num_agents, 2]
            
            # Create edges between agents that are within a certain radius
            max_radius = 50.0  # meters, can be tuned
            
            for i in range(num_agents):
                for j in range(num_agents):
                    if i != j:  # Don't connect agent to itself
                        pos_i = agent_positions[i]
                        pos_j = agent_positions[j]
                        
                        # Calculate Euclidean distance
                        dist = torch.norm(pos_i - pos_j)
                        
                        if dist < max_radius:
                            # Connect these agents
                            src_idx = b * num_agents + i
                            dst_idx = b * num_agents + j
                            
                            edge_indices.append([src_idx, dst_idx])
                            
                            # Edge features: relative position and distance
                            rel_pos = pos_j - pos_i
                            edge_attrs.append([rel_pos[0], rel_pos[1], dist])
    
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    
    return node_feats, edge_index, edge_attr