import torch
import torch.nn as nn
import torch.nn.functional as F

class DirectionAwareGRUEncoder(nn.Module):
    """GRU encoder with explicit handling of motion direction, heading and object type"""
    
    def __init__(self, input_dim=6, hidden_dim=128, num_layers=2, num_object_types=10):
        super(DirectionAwareGRUEncoder, self).__init__()
        
        # Object type embedding
        self.object_embedding = nn.Embedding(num_object_types, hidden_dim // 4)
        
        # Heading encoder
        self.heading_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.ReLU()
        )
        
        # Standard GRU layer
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Direction and velocity encoder
        self.direction_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim // 4),  # Takes last 2 velocities (4 values)
            nn.ReLU()
        )
        
        # Fusion layer to combine GRU output with direction, heading and object type features
        self.fusion = nn.Linear(hidden_dim + hidden_dim // 4 * 3, hidden_dim)
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
    
    def forward(self, x):
        """
        Forward pass through the encoder
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            
        Returns:
            output: Full sequence of outputs
            hidden: Final hidden states with direction, heading and object type information incorporated
        """
        batch_size = x.size(0)
        
        # Extract features
        positions = x[:, :, :2]  # [batch_size, seq_len, 2]
        velocities = x[:, :, 2:4]  # [batch_size, seq_len, 2]
        heading = x[:, -1:, 4:5]  # Latest heading [batch_size, 1, 1]
        
        # Extract object type (safely handle potential out-of-bounds values)
        object_type = x[:, -1, 5].long()  # Use the latest frame's object type
        object_type = torch.clamp(object_type, 0, 9)  # Ensure it's in valid range
        
        # Calculate the last two velocities for direction information
        if x.size(1) >= 3:
            vel1 = velocities[:, -1, :]  # Last velocity
            vel2 = velocities[:, -2, :]  # Second-to-last velocity
            dir_features = torch.cat([vel1, vel2], dim=1)  # [batch_size, 4]
        else:
            # Fallback if sequence is too short
            dir_features = torch.zeros(batch_size, 4, device=x.device)
        
        # Apply GRU
        output, hidden = self.gru(x)
        
        # Encode direction, heading and object type information
        direction_features = self.direction_encoder(dir_features)
        heading_features = self.heading_encoder(heading).squeeze(1)  # [batch_size, hidden_dim//4]
        object_features = self.object_embedding(object_type)  # [batch_size, hidden_dim//4]
        
        # Combine all features with the hidden state
        enhanced_hidden = []
        for layer_idx in range(self.num_layers):
            # Concatenate features with hidden state for this layer
            layer_hidden = hidden[layer_idx]  # [batch_size, hidden_dim]
            combined = torch.cat([
                layer_hidden, 
                direction_features, 
                heading_features, 
                object_features
            ], dim=1)
            
            # Fuse them together
            enhanced_layer = self.fusion(combined)
            enhanced_hidden.append(enhanced_layer)
        
        # Stack back into the expected shape for hidden state
        enhanced_hidden = torch.stack(enhanced_hidden, dim=0)
        
        return output, enhanced_hidden


class PhysicsAwareGRUDecoder(nn.Module):
    """GRU decoder with physics-aware trajectory generation"""
    
    def __init__(self, input_dim=2, hidden_dim=128, output_dim=2, num_layers=2, num_object_types=10):
        super(PhysicsAwareGRUDecoder, self).__init__()
        
        # Object type-specific physics parameters
        self.object_params = nn.Embedding(num_object_types, hidden_dim // 4)
        
        # Velocity input layer - FIXED to match actual input shape
        self.velocity_encoder = nn.Linear(input_dim, input_dim)
        
        # Heading encoder
        self.heading_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.ReLU()
        )
        
        # GRU layer - FIXED input size calculation
        self.gru = nn.GRU(
            input_size=input_dim + input_dim + hidden_dim // 2,  # Position + encoded velocity + object & heading
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Output projections for position, velocity and heading
        self.position_out = nn.Linear(hidden_dim, output_dim)
        self.velocity_out = nn.Linear(hidden_dim, output_dim)
        self.heading_out = nn.Linear(hidden_dim, 1)
        
        # Physics-based refinement - object type specific
        self.physics_refinement = nn.Sequential(
            nn.Linear(hidden_dim + output_dim * 2 + 1 + hidden_dim // 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x, hidden, prev_velocity=None, heading=None, object_type=None):
        """
        Forward pass through the decoder
        
        Args:
            x: Input tensor of shape [batch_size, 1, input_dim] (just position)
            hidden: Hidden state from encoder or previous step
            prev_velocity: Previous velocity prediction
            heading: Current heading in radians
            object_type: Object type (integer index)
            
        Returns:
            prediction: Position prediction
            hidden_state: Updated hidden state
            velocity: Velocity prediction
            new_heading: Updated heading
        """
        batch_size = x.size(0)
        
        # Initialize velocity if not provided
        if prev_velocity is None:
            prev_velocity = torch.zeros_like(x)
        
        # Initialize heading if not provided (assume straight ahead)
        if heading is None:
            heading = torch.zeros(batch_size, 1, 1, device=x.device)
            
        # Encode velocity - FIXED to properly handle the tensor shape
        # prev_velocity shape is [batch_size, 1, input_dim]
        velocity_features = self.velocity_encoder(prev_velocity.reshape(batch_size, 1, -1))
        
        # Encode heading
        heading_features = self.heading_encoder(heading).view(batch_size, 1, -1)  # Reshape to [batch_size, 1, hidden_dim//4]
        
        # Get object type parameters
        if object_type is None:
            # Default to vehicle (type 0) if not specified
            object_type = torch.zeros(batch_size, dtype=torch.long, device=x.device)
        
        # Ensure object_type is in valid range
        object_type = torch.clamp(object_type, 0, 9)
        object_features = self.object_params(object_type).unsqueeze(1)
        
        # Concatenate all input features
        combined_input = torch.cat([
            x,
            velocity_features,
            heading_features,  # Now properly shaped as [batch_size, 1, hidden_dim//4]
            object_features
        ], dim=2)
        
        # Apply GRU
        output, hidden_state = self.gru(combined_input, hidden)
        
        # Generate position, velocity and heading predictions
        position_pred = self.position_out(output)
        velocity_pred = self.velocity_out(output)
        heading_pred = self.heading_out(output)
        
        # Refine position prediction using physics and object type
        physics_input = torch.cat([
            output,
            position_pred,
            velocity_pred,
            heading_pred,
            object_features  # Already has shape [batch_size, 1, hidden_dim//4]
        ], dim=2)
        
        refined_position = self.physics_refinement(physics_input)
        
        return refined_position, hidden_state, velocity_pred, heading_pred


class DirectionAwareSeq2SeqGRUModel(nn.Module):
    """Sequence-to-sequence GRU model with direction awareness and physics constraints"""
    
    def __init__(self, input_dim=6, hidden_dim=128, output_seq_len=60, output_dim=2, num_layers=2, num_object_types=10):
        super(DirectionAwareSeq2SeqGRUModel, self).__init__()
        
        # Direction-aware encoder and physics-aware decoder
        self.encoder = DirectionAwareGRUEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_object_types=num_object_types
        )
        
        self.decoder = PhysicsAwareGRUDecoder(
            input_dim=output_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            num_object_types=num_object_types
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
        
        # Extract last heading and object type for decoder initialization
        last_heading = ego_history[:, -1:, 4:5]
        object_type = ego_history[:, -1, 5].long()
        
        # Encode the history sequence
        _, hidden = self.encoder(ego_history)
        
        # Determine if we're in training or inference mode
        use_teacher_forcing = 'future' in data and torch.rand(1).item() < teacher_forcing_ratio
        
        # Initialize output container
        predictions = torch.zeros(batch_size, self.output_seq_len, self.output_dim, 
                                 device=history.device)
        
        # Initialize decoder input with the last position from history (x,y coordinates)
        decoder_input = ego_history[:, -1:, :self.output_dim]
        
        # Calculate initial velocity from last two history positions
        if ego_history.shape[1] >= 2:
            prev_velocity = ego_history[:, -1:, 2:4]  # Use the actual velocity from history
        else:
            prev_velocity = torch.zeros_like(decoder_input)
        
        # Current heading
        curr_heading = last_heading
        
        # Iteratively decode the sequence
        for t in range(self.output_seq_len):
            # Get prediction for current timestep
            position_pred, hidden, velocity_pred, heading_pred = self.decoder(
                decoder_input, hidden, prev_velocity, curr_heading, object_type
            )
            
            # Store position prediction
            predictions[:, t:t+1, :] = position_pred
            
            # Update decoder input for next timestep
            if use_teacher_forcing and t < self.output_seq_len - 1 and 'future' in data:
                # Use ground truth as next input
                next_position = data['future'][:, t:t+1, :]
                
                # Calculate velocity based on ground truth
                if t > 0:
                    prev_velocity = next_position - data['future'][:, t-1:t, :]
                else:
                    prev_velocity = next_position - decoder_input
                
                # Update position
                decoder_input = next_position
                
                # Update heading based on movement direction 
                dx = prev_velocity[:, 0, 0]
                dy = prev_velocity[:, 0, 1]
                heading = torch.atan2(dy, dx).unsqueeze(-1).unsqueeze(-1)
                curr_heading = heading
            else:
                # Use current prediction as next input
                prev_velocity = velocity_pred
                decoder_input = position_pred
                curr_heading = heading_pred
        
        return predictions

    def compute_losses(self, predictions, targets, history=None):
        """
        Compute multiple loss components including direction consistency
        
        Args:
            predictions: Predicted trajectories [batch_size, seq_len, 2]
            targets: Ground truth trajectories [batch_size, seq_len, 2]
            history: Historical trajectories [batch_size, seq_len, feature_dim]
            
        Returns:
            Dictionary of loss components and total loss
        """
        # Main displacement loss
        mse_loss = F.mse_loss(predictions, targets)
        
        # Calculate direction loss (consistency between steps)
        direction_loss = 0.0
        if predictions.size(1) > 1:
            # Calculate velocities for predictions and targets
            pred_velocities = predictions[:, 1:] - predictions[:, :-1]
            target_velocities = targets[:, 1:] - targets[:, :-1]
            
            # Normalize to get directions
            pred_vel_norm = torch.norm(pred_velocities, dim=2, keepdim=True)
            target_vel_norm = torch.norm(target_velocities, dim=2, keepdim=True)
            
            # Avoid division by zero
            eps = 1e-8
            pred_directions = pred_velocities / (pred_vel_norm + eps)
            target_directions = target_velocities / (target_vel_norm + eps)
            
            # Direction loss (1 - cos similarity)
            cos_sim = torch.sum(pred_directions * target_directions, dim=2)
            direction_loss = torch.mean(1.0 - cos_sim)
        
        # Calculate initial direction consistency with history
        initial_direction_loss = 0.0
        if history is not None and history.size(1) >= 2 and predictions.size(1) >= 1:
            # Extract the heading from history
            last_heading = history[:, -1, 4]
            
            # Last history velocity
            last_history_vel = history[:, -1, 2:4]  # Use actual velocity feature
            
            # Alternatively, use the position difference
            if torch.all(torch.abs(last_history_vel) < 1e-6):
                last_history_vel = history[:, -1, :2] - history[:, -2, :2]
            
            # First prediction velocity
            first_pred_vel = predictions[:, 0, :] - history[:, -1, :2]
            
            # Normalize
            last_hist_vel_norm = torch.norm(last_history_vel, dim=1, keepdim=True)
            first_pred_vel_norm = torch.norm(first_pred_vel, dim=1, keepdim=True)
            
            # Avoid division by zero
            eps = 1e-8
            last_hist_dir = last_history_vel / (last_hist_vel_norm + eps)
            first_pred_dir = first_pred_vel / (first_pred_vel_norm + eps)
            
            # Initial direction loss (1 - cos similarity)
            init_cos_sim = torch.sum(last_hist_dir * first_pred_dir, dim=1)
            initial_direction_loss = torch.mean(1.0 - init_cos_sim)
            
            # Heading consistency (if heading is available)
            # Convert heading to direction vector and compare with prediction direction
            heading_x = torch.cos(last_heading)
            heading_y = torch.sin(last_heading)
            heading_vec = torch.stack([heading_x, heading_y], dim=1)
            
            # Heading direction loss
            heading_cos_sim = torch.sum(heading_vec * first_pred_dir, dim=1)
            heading_direction_loss = torch.mean(1.0 - heading_cos_sim)
            
            # Add heading direction loss to initial direction loss
            initial_direction_loss = 0.5 * initial_direction_loss + 0.5 * heading_direction_loss
        
        # Final loss - weighted components
        total_loss = mse_loss + 0.5 * direction_loss + 2.0 * initial_direction_loss
        
        return {
            'mse_loss': mse_loss,
            'direction_loss': direction_loss,
            'initial_direction_loss': initial_direction_loss,
            'total_loss': total_loss
        }