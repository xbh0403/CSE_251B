import torch
import torch_geometric

def process_batch_sparse(batch_data, max_radius=50.0):
    """
    Process batch using sparse tensor operations for maximum speed
    
    Args:
        batch_data: tensor of shape [batch_size, num_agents, time_steps, feature_dim]
        max_radius: Maximum radius for connecting agents
        
    Returns:
        node_feats: Node features for GNN
        edge_index: Edge connections (which nodes are connected)
        edge_attr: Edge attributes (features of the connections)
    """
    
    batch_size, num_agents, seq_len, feat_dim = batch_data.shape
    
    # Create node features
    node_feats = batch_data.reshape(batch_size * num_agents, seq_len, feat_dim)
    
    # Use only the last timestep for building the graph
    positions = batch_data[:, :, -1, :2]  # [batch_size, num_agents, 2]
    
    all_edges = []
    all_edge_attrs = []
    
    for b in range(batch_size):
        # Get positions for this batch
        batch_positions = positions[b]  # [num_agents, 2]
        
        # Filter out any NaN positions (agents that don't exist)
        valid_mask = ~torch.isnan(batch_positions).any(dim=1)
        valid_positions = batch_positions[valid_mask]
        valid_indices = torch.where(valid_mask)[0]
        
        # If less than 2 agents, skip
        if len(valid_positions) < 2:
            continue
        
        # Create radius graph using torch_geometric
        batch_edge_index = torch_geometric.nn.radius_graph(
            valid_positions, 
            r=max_radius,
            batch=None,
            loop=False,
            max_num_neighbors=num_agents - 1,
            flow='source_to_target'
        )
        
        # Map back to original indices
        src = valid_indices[batch_edge_index[0]] + b * num_agents
        dst = valid_indices[batch_edge_index[1]] + b * num_agents
        
        # Create edge attributes
        for s, d in zip(src, dst):
            # Get original positions
            orig_s = s % num_agents
            orig_d = d % num_agents
            
            pos_s = batch_positions[orig_s]
            pos_d = batch_positions[orig_d]
            
            # Calculate edge features
            rel_pos = pos_d - pos_s
            dist = torch.norm(rel_pos)
            
            all_edge_attrs.append([rel_pos[0].item(), rel_pos[1].item(), dist.item()])
        
        # Store edges
        all_edges.append(torch.stack([src, dst], dim=0))
    
    # Combine all edges
    if all_edges:
        edge_index = torch.cat(all_edges, dim=1)
        edge_attr = torch.tensor(all_edge_attrs, dtype=torch.float)
    else:
        # Empty graph
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 3), dtype=torch.float)
    
    return node_feats, edge_index, edge_attr


def process_batch_radius_pure_torch(batch_data, max_radius=50.0):
    """
    Process batch using radius with pure PyTorch (no torch_geometric)
    
    Args:
        batch_data: tensor of shape [batch_size, num_agents, time_steps, feature_dim]
        max_radius: Maximum connection radius
        
    Returns:
        node_feats: Node features for GNN
        edge_index: Edge connections
        edge_attr: Edge attributes
    """
    device = batch_data.device
    batch_size, num_agents, seq_len, feat_dim = batch_data.shape
    
    # Create node features
    node_feats = batch_data.reshape(batch_size * num_agents, seq_len, feat_dim)
    
    # Use last timestep for graph construction
    positions = batch_data[:, :, -1, :2]  # [batch_size, num_agents, 2]
    
    all_sources = []
    all_targets = []
    all_edge_attrs = []
    
    for b in range(batch_size):
        # Get positions for this batch
        batch_positions = positions[b]  # [num_agents, 2]
        
        # Skip NaN positions
        valid_mask = ~torch.isnan(batch_positions).any(dim=1)
        if valid_mask.sum() < 2:  # Need at least 2 valid agents
            continue
            
        valid_positions = batch_positions[valid_mask]
        valid_indices = torch.where(valid_mask)[0]
        
        # Compute pairwise distances
        dist_matrix = torch.cdist(valid_positions, valid_positions)
        
        # Find pairs within radius (excluding self-connections)
        mask = (dist_matrix < max_radius) & ~torch.eye(
            valid_indices.size(0), dtype=torch.bool, device=device
        )
        
        # Get source and target indices
        rows, cols = torch.where(mask)
        
        # Skip if no connections
        if rows.size(0) == 0:
            continue
            
        # Map to original indices and adjust for batch
        source = valid_indices[rows] + b * num_agents
        target = valid_indices[cols] + b * num_agents
        
        # Compute edge attributes
        source_pos = valid_positions[rows]
        target_pos = valid_positions[cols]
        
        rel_pos = target_pos - source_pos
        dist = dist_matrix[rows, cols].unsqueeze(1)
        
        all_sources.append(source)
        all_targets.append(target)
        all_edge_attrs.append(torch.cat([rel_pos, dist], dim=1))
    
    # Combine results
    if all_sources:
        sources = torch.cat(all_sources)
        targets = torch.cat(all_targets)
        edge_index = torch.stack([sources, targets], dim=0)
        edge_attr = torch.cat(all_edge_attrs, dim=0)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        edge_attr = torch.zeros((0, 3), dtype=torch.float, device=device)
    
    return node_feats, edge_index, edge_attr