import torch

def process_batch_radius(batch_data, max_radius=50.0):
    """
    Process batch using radius
    
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

def compute_constant_velocity(history_data):
    """
    Compute constant velocity baseline predictions
    
    Args:
        history_data: tensor of shape [batch_size, num_agents, seq_len, feature_dim]
        
    Returns:
        predictions: tensor of shape [batch_size, output_seq_len=60, 2]
    """
    batch_size, num_agents, seq_len, feat_dim = history_data.shape
    device = history_data.device
    
    # Extract focal agent positions and velocities
    focal_positions = history_data[:, 0, :, :2]  # [batch_size, seq_len, 2]
    focal_velocities = history_data[:, 0, :, 2:4]  # [batch_size, seq_len, 2]
    
    # Use the most recent velocity from history
    last_velocity = focal_velocities[:, -1, :]  # [batch_size, 2]
    last_position = focal_positions[:, -1, :]  # [batch_size, 2]
    
    # Generate future positions using constant velocity
    future_timesteps = 60  # Number of future timesteps to predict
    time_delta = 0.1  # Assuming 10Hz sampling rate
    
    # Initialize predictions tensor
    predictions = torch.zeros((batch_size, future_timesteps, 2), device=device)
    
    # Generate predictions
    for t in range(future_timesteps):
        t_future = (t + 1) * time_delta  # Time from last observed position
        predictions[:, t, :] = last_position + last_velocity * t_future
    
    return predictions

def compute_physics_features(history_data):
    """
    Compute physics-based features for trajectory prediction
    
    Args:
        history_data: tensor of shape [batch_size, num_agents, seq_len, feature_dim]
        
    Returns:
        physics_features: tensor of shape [batch_size, num_agents, 6]
            contains: [avg_velocity_x, avg_velocity_y, accel_x, accel_y, 
                      heading, angular_velocity]
    """
    batch_size, num_agents, seq_len, feat_dim = history_data.shape
    device = history_data.device
    
    # Extract positions and velocities
    positions = history_data[..., :2]  # [batch_size, num_agents, seq_len, 2]
    velocities = history_data[..., 2:4]  # [batch_size, num_agents, seq_len, 2]
    
    # Calculate average velocity over last 5 timesteps
    recent_vel = velocities[:, :, -5:, :]  # [batch_size, num_agents, 5, 2]
    avg_velocity = torch.nanmean(recent_vel, dim=2)  # [batch_size, num_agents, 2]
    
    # Calculate acceleration (velocity change over last 10 timesteps)
    if seq_len >= 10:
        vel_start = velocities[:, :, -10, :]  # [batch_size, num_agents, 2]
        vel_end = velocities[:, :, -1, :]  # [batch_size, num_agents, 2]
        accel = (vel_end - vel_start) / 1.0  # Assuming 10Hz, so 10 frames = 1 second
    else:
        # If we don't have enough history, use the difference between last two frames
        vel_start = velocities[:, :, -2, :]  # [batch_size, num_agents, 2]
        vel_end = velocities[:, :, -1, :]  # [batch_size, num_agents, 2]
        accel = (vel_end - vel_start) / 0.1  # 1 frame = 0.1 seconds
    
    # Calculate heading (direction of travel)
    heading = torch.atan2(avg_velocity[:, :, 1], avg_velocity[:, :, 0])  # [batch_size, num_agents]
    
    # Calculate angular velocity (change in heading)
    if seq_len >= 10:
        pos_diff_start = positions[:, :, -10:-9, :] - positions[:, :, -11:-10, :]
        pos_diff_end = positions[:, :, -1:, :] - positions[:, :, -2:-1, :]
        
        heading_start = torch.atan2(pos_diff_start[:, :, 0, 1], pos_diff_start[:, :, 0, 0])
        heading_end = torch.atan2(pos_diff_end[:, :, 0, 1], pos_diff_end[:, :, 0, 0])
        
        # Ensure the angle difference is between -pi and pi
        heading_diff = heading_end - heading_start
        heading_diff = torch.atan2(torch.sin(heading_diff), torch.cos(heading_diff))
        
        angular_velocity = heading_diff / 1.0  # 10 frames = 1 second
    else:
        angular_velocity = torch.zeros((batch_size, num_agents), device=device)
    
    # Combine physics features
    physics_features = torch.cat([
        avg_velocity,  # [batch_size, num_agents, 2]
        accel,         # [batch_size, num_agents, 2]
        heading.unsqueeze(-1),         # [batch_size, num_agents, 1]
        angular_velocity.unsqueeze(-1)  # [batch_size, num_agents, 1]
    ], dim=-1)
    
    return physics_features