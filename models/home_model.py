import torch
import torch.nn as nn
import torch.nn.functional as F

class TypeSpecificEncoder(nn.Module):
    """Encoder with type-specific parameters for different agent types"""
    
    def __init__(self, input_dim=6, hidden_dim=128, num_layers=2, num_types=10):
        super(TypeSpecificEncoder, self).__init__()
        self.num_types = num_types
        
        # Type embedding layer
        self.type_embedding = nn.Embedding(num_types, 16)
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim + 16, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Type-specific GRU encoders
        self.encoders = nn.ModuleDict({
            # Core types with specific encoders
            'vehicle': nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True),
            'pedestrian': nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True),
            'cyclist': nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True),
            'bus': nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True),
            # Default encoder for other types
            'default': nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        })
        
        # Type to encoder mapping
        self.type_to_encoder = {
            0: 'vehicle',
            1: 'pedestrian',
            2: 'motorcyclist',
            3: 'cyclist',
            4: 'bus',
            5: 'default',  # static
            6: 'default',  # background
            7: 'default',  # construction
            8: 'default',  # riderless_bicycle
            9: 'default'   # unknown
        }
        
        self.hidden_dim = hidden_dim
    
    def forward(self, x, obj_type):
        """
        Forward pass with type-specific processing
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            obj_type: Object type indices [batch_size]
            
        Returns:
            outputs: Encoded outputs [batch_size, seq_len, hidden_dim]
            hidden: Final hidden state [num_layers, batch_size, hidden_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Get type embeddings and expand to match sequence
        type_embed = self.type_embedding(obj_type)  # [batch_size, 16]
        type_embed = type_embed.unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, 16]
        
        # Concatenate type embedding with input
        x_with_type = torch.cat([x, type_embed], dim=2)  # [batch_size, seq_len, input_dim+16]
        
        # Extract features
        features = self.feature_extractor(x_with_type)  # [batch_size, seq_len, hidden_dim]
        
        # Process each sample with its type-specific encoder
        outputs_list = []
        hidden_list = []
        
        # Group samples by type for efficient processing
        for type_idx in range(self.num_types):
            # Get indices of samples with this type
            indices = (obj_type == type_idx).nonzero(as_tuple=True)[0]
            
            if len(indices) == 0:
                continue
                
            # Get encoder for this type
            encoder_key = self.type_to_encoder[type_idx]
            encoder = self.encoders[encoder_key]
            
            # Select features for these samples
            type_features = features[indices]
            
            # Apply type-specific encoder
            type_outputs, type_hidden = encoder(type_features)
            
            # Store outputs and hidden states
            outputs_list.append((indices, type_outputs))
            hidden_list.append((indices, type_hidden))
        
        # Combine outputs and hidden states
        outputs = torch.zeros(batch_size, seq_len, self.hidden_dim, device=x.device)
        hidden = torch.zeros(self.encoders['default'].num_layers, batch_size, self.hidden_dim, device=x.device)
        
        for indices, type_outputs in outputs_list:
            outputs[indices] = type_outputs
            
        for indices, type_hidden in hidden_list:
            hidden[:, indices] = type_hidden
        
        return outputs, hidden


class HeatmapDecoder(nn.Module):
    """Decoder with heatmap output for multimodal trajectory prediction"""
    
    def __init__(self, input_dim=2, hidden_dim=128, output_seq_len=60, 
                 grid_size=64, grid_range=[-50, 50], num_layers=2):
        super(HeatmapDecoder, self).__init__()
        
        # Store params
        self.hidden_dim = hidden_dim
        self.output_seq_len = output_seq_len
        self.grid_size = grid_size
        self.grid_range = grid_range
        self.grid_cell_size = (grid_range[1] - grid_range[0]) / grid_size
        
        # Decoder GRU
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Trajectory heatmap generator
        self.heatmap_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_seq_len * 2)  # x,y for each timestep
        )
        
        # Confidence score for each timestep
        self.confidence_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_seq_len)
        )
        
        # Heatmap predictor for each future timestep (shared weights)
        self.heatmap_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, grid_size * grid_size)  # Flattened heatmap
        )
    
    def forward(self, x, hidden, num_samples=6):
        """
        Forward pass with heatmap-based multimodal trajectory prediction
        
        Args:
            x: Input tensor [batch_size, 1, input_dim] - just one timestep
            hidden: Hidden state from encoder [num_layers, batch_size, hidden_dim]
            num_samples: Number of trajectory samples to generate
            
        Returns:
            predictions: Trajectory predictions [batch_size, num_samples, output_seq_len, 2]
            confidences: Prediction confidences [batch_size, num_samples]
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Apply GRU
        _, hidden_state = self.gru(x, hidden)
        
        # Use last layer hidden state
        last_hidden = hidden_state[-1]  # [batch_size, hidden_dim]
        
        # Generate trajectory parameters
        traj_params = self.heatmap_generator(last_hidden)  # [batch_size, output_seq_len*2]
        traj_params = traj_params.view(batch_size, self.output_seq_len, 2)  # [batch_size, output_seq_len, 2]
        
        # Generate confidence scores
        confidence = self.confidence_estimator(last_hidden)  # [batch_size, output_seq_len]
        
        # Generate heatmaps for each timestep
        heatmaps = []
        for t in range(self.output_seq_len):
            # Get hidden representation for this timestep (could be enhanced further)
            timestep_hidden = last_hidden + 0.1 * torch.randn_like(last_hidden) * (t / self.output_seq_len)
            
            # Generate heatmap
            heatmap = self.heatmap_predictor(timestep_hidden)  # [batch_size, grid_size*grid_size]
            heatmap = heatmap.view(batch_size, self.grid_size, self.grid_size)  # [batch_size, grid_size, grid_size]
            heatmap = F.softmax(heatmap.view(batch_size, -1), dim=1).view(batch_size, self.grid_size, self.grid_size)
            
            heatmaps.append(heatmap)
        
        # Sample trajectories from heatmaps
        predictions = torch.zeros(batch_size, num_samples, self.output_seq_len, 2, device=device)
        confidences = torch.zeros(batch_size, num_samples, device=device)
        
        for b in range(batch_size):
            # Use deterministic and heatmap samples
            
            # First trajectory is the mean prediction (highest probability)
            predictions[b, 0] = traj_params[b]
            confidences[b, 0] = torch.mean(confidence[b])
            
            # Other trajectories are sampled from heatmaps
            for s in range(1, num_samples):
                sample_points = []
                
                # Sample each timestep
                for t, heatmap in enumerate(heatmaps):
                    # Convert heatmap to probability distribution
                    prob_dist = heatmap[b].view(-1)
                    
                    # Sample from distribution
                    try:
                        idx = torch.multinomial(prob_dist, 1).item()
                        
                        # Convert flat index to 2D coordinates
                        y_idx = idx // self.grid_size
                        x_idx = idx % self.grid_size
                        
                        # Convert indices to actual coordinates
                        x = self.grid_range[0] + x_idx * self.grid_cell_size + self.grid_cell_size / 2
                        y = self.grid_range[0] + y_idx * self.grid_cell_size + self.grid_cell_size / 2
                        
                        # Add point
                        sample_points.append([x, y])
                    except:
                        # Fallback if sampling fails
                        sample_points.append([traj_params[b, t, 0].item(), traj_params[b, t, 1].item()])
                
                # Store sampled trajectory
                predictions[b, s] = torch.tensor(sample_points, device=device)
                confidences[b, s] = torch.mean(confidence[b]) * (0.9 ** s)  # Lower confidence for each sample
        
        return predictions, confidences


class HOMEModel(nn.Module):
    """HOME (Heatmap Output for future Motion Estimation) model"""
    
    def __init__(self, input_dim=6, hidden_dim=128, output_seq_len=60, 
                 grid_size=64, grid_range=[-50, 50], num_samples=6, num_layers=2):
        super(HOMEModel, self).__init__()
        
        # Type-specific encoder
        self.encoder = TypeSpecificEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )
        
        # Heatmap-based decoder
        self.decoder = HeatmapDecoder(
            input_dim=2,  # x, y position
            hidden_dim=hidden_dim,
            output_seq_len=output_seq_len,
            grid_size=grid_size,
            grid_range=grid_range,
            num_layers=num_layers
        )
        
        # Parameters
        self.output_seq_len = output_seq_len
        self.hidden_dim = hidden_dim
        self.num_samples = num_samples
        
        # Physics constraints by object type (in normalized coordinates)
        # Values are [max_speed, max_accel, max_angular_vel]
        self.physics_constraints = {
            0: [1.5348, 1.0870, 15.7080],  # vehicle
            1: [0.2467, 0.3594, 15.7080],  # pedestrian
            2: [1.7693, 1.2837, 19.7491],  # motorcyclist
            3: [0.9361, 0.8407, 15.7080],  # cyclist
            4: [1.6763, 2.9826, 18.8791],  # bus
            5: [0.2475, 1.5070, 27.5866],  # static
            6: [0.3499, 2.0938, 29.1178],  # background
            7: [0.2001, 1.1641, 27.2448],  # construction
            8: [0.3446, 2.2755, 28.8118],  # riderless_bicycle
            9: [1.0772, 6.5438, 29.3093],  # unknown
        }
    
    def forward(self, data, teacher_forcing_ratio=0.5):
        """
        Forward pass through the network
        
        Args:
            data: Dictionary containing:
                - history: Shape [batch_size, num_agents, seq_len, feature_dim]
                - future: Shape [batch_size, output_seq_len, output_dim] (only in training)
            teacher_forcing_ratio: Probability of using teacher forcing (0-1)
                
        Returns:
            predictions: Multiple trajectory predictions [batch_size, num_samples, output_seq_len, 2]
            confidences: Confidence values for each prediction [batch_size, num_samples]
        """
        # Extract history data
        history = data['history']
        batch_size = history.shape[0]
        
        # Extract ego agent only
        ego_history = history[:, 0, :, :]
        
        # Get object type for ego agent
        # Take the most common non-zero type value across the sequence
        ego_type = ego_history[:, 0, 5].long()  # Just use first timestep's type
        
        # Handle invalid types (negative or too large)
        ego_type = torch.clamp(ego_type, min=0, max=9)
        
        # Encode the history sequence with type-specific encoder
        _, hidden = self.encoder(ego_history, ego_type)
        
        # Initialize decoder input with the last position from history
        decoder_input = ego_history[:, -1:, :2]  # [batch_size, 1, 2]
        
        # Generate predictions
        predictions, confidences = self.decoder(decoder_input, hidden, self.num_samples)
        
        # Apply physics constraints based on object type
        predictions = self.apply_physics_constraints(predictions, ego_type)
        
        # If in inference mode or not using teacher forcing, return all predictions
        if 'future' not in data or torch.rand(1).item() >= teacher_forcing_ratio:
            return predictions, confidences
        
        # If using teacher forcing, update training signal
        # For training, we only use the highest confidence prediction (index 0)
        return predictions[:, 0], confidences[:, 0]
    
    def apply_physics_constraints(self, predictions, obj_types):
        """
        Apply physics constraints to predictions based on object types
        
        Args:
            predictions: Trajectory predictions [batch_size, num_samples, output_seq_len, 2]
            obj_types: Object type indices [batch_size]
            
        Returns:
            constrained_predictions: Physics-constrained predictions
        """
        batch_size, num_samples, seq_len, _ = predictions.shape
        constrained_predictions = predictions.clone()
        device = predictions.device
        
        for b in range(batch_size):
            obj_type = obj_types[b].item()
            constraints = self.physics_constraints.get(obj_type, self.physics_constraints[9])
            
            # Convert constraints to tensors on the correct device
            max_speed = torch.tensor(constraints[0], device=device)
            max_accel = torch.tensor(constraints[1], device=device)
            max_angular_vel = torch.tensor(constraints[2], device=device)
            dt = torch.tensor(0.1, device=device)  # 10Hz
            
            for s in range(num_samples):
                trajectory = constrained_predictions[b, s]
                
                # Apply constraints sequentially for each timestep
                for t in range(1, seq_len):
                    if t == 1:
                        # For first prediction, limit based on last history point
                        # Limit speed
                        displacement = trajectory[t] - trajectory[t-1]
                        speed = torch.norm(displacement) / dt
                        
                        if speed > max_speed:
                            # Scale back to max speed
                            trajectory[t] = trajectory[t-1] + displacement * (max_speed * dt / speed)
                    else:
                        # For subsequent predictions, consider previous predictions
                        # Calculate current velocity
                        curr_vel = (trajectory[t] - trajectory[t-1]) / dt
                        prev_vel = (trajectory[t-1] - trajectory[t-2]) / dt
                        
                        # Calculate acceleration
                        accel_vec = (curr_vel - prev_vel) / dt
                        accel_mag = torch.norm(accel_vec)
                        
                        # Limit acceleration
                        if accel_mag > max_accel:
                            # Scale acceleration
                            scaled_accel = accel_vec * (max_accel / accel_mag)
                            # Update velocity based on limited acceleration
                            new_vel = prev_vel + scaled_accel * dt
                            # Update position
                            trajectory[t] = trajectory[t-1] + new_vel * dt
                        
                        # Limit angular velocity (turning rate)
                        if t >= 2:
                            # Calculate directions
                            dir_curr = F.normalize(trajectory[t] - trajectory[t-1], dim=0)
                            dir_prev = F.normalize(trajectory[t-1] - trajectory[t-2], dim=0)
                            
                            # Calculate angle between directions (dot product)
                            cos_angle = torch.sum(dir_curr * dir_prev)
                            # Clamp to avoid numerical issues
                            cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
                            angle = torch.acos(cos_angle)
                            
                            # Check if angle exceeds max angular velocity
                            if angle > max_angular_vel * dt:
                                # Calculate rotation matrix for max allowed rotation
                                cos_max = torch.cos(max_angular_vel * dt)
                                sin_max = torch.sin(max_angular_vel * dt)
                                
                                # Determine rotation direction (same as original)
                                cross_prod = dir_prev[0] * dir_curr[1] - dir_prev[1] * dir_curr[0]
                                sign = 1 if cross_prod >= 0 else -1
                                
                                # Create rotation matrix
                                rot_matrix = torch.tensor([
                                    [cos_max, -sign * sin_max],
                                    [sign * sin_max, cos_max]
                                ], device=device)
                                
                                # Apply rotation to previous direction
                                new_dir = torch.matmul(rot_matrix, dir_prev.unsqueeze(1)).squeeze(1)
                                
                                # Calculate new position
                                displacement = torch.norm(trajectory[t] - trajectory[t-1])
                                trajectory[t] = trajectory[t-1] + new_dir * displacement
        
        return constrained_predictions

    def compute_loss(self, predictions, confidences, ground_truth, obj_types):
        """
        Compute loss for HOME model
        
        Args:
            predictions: Predictions [batch_size, num_samples, output_seq_len, 2] or [batch_size, output_seq_len, 2]
            confidences: Confidence values [batch_size, num_samples] or None
            ground_truth: Ground truth [batch_size, output_seq_len, 2]
            obj_types: Object types [batch_size]
            
        Returns:
            loss: Total loss value
        """
        # Handle both training and inference outputs
        if len(predictions.shape) == 4:
            # Multi-modal predictions - use winner-takes-all approach
            batch_size, num_samples, seq_len, _ = predictions.shape
            
            # Calculate ADE for each sample
            ade_per_sample = []
            for s in range(num_samples):
                sample_pred = predictions[:, s]
                # ADE is average displacement error across sequence
                displacement = torch.norm(sample_pred - ground_truth, dim=2)  # [batch_size, seq_len]
                ade = torch.mean(displacement, dim=1)  # [batch_size]
                ade_per_sample.append(ade)
            
            # Stack ADEs for all samples
            all_ades = torch.stack(ade_per_sample, dim=1)  # [batch_size, num_samples]
            
            # Find best prediction (minimum ADE) for each batch item
            min_ade, min_idx = torch.min(all_ades, dim=1)  # [batch_size]
            
            # Compute loss using minimum ADE
            prediction_loss = torch.mean(min_ade)
            
            # Add confidence loss (encourage higher confidence for better predictions)
            conf_loss = torch.tensor(0.0, device=predictions.device)
            if confidences is not None:
                # For each batch, create target: 1 for best sample, 0 for others
                target_conf = torch.zeros_like(confidences)
                for b in range(batch_size):
                    target_conf[b, min_idx[b]] = 1.0
                
                # Binary cross entropy loss
                conf_loss = F.binary_cross_entropy_with_logits(confidences, target_conf)
            
            # Add physics-based regularization loss
            physics_loss = self.compute_physics_loss(predictions, obj_types)
            
            # Total loss
            loss = prediction_loss + 0.2 * conf_loss + 0.1 * physics_loss
        else:
            # Single trajectory prediction (during training with teacher forcing)
            # Regular MSE loss
            prediction_loss = F.mse_loss(predictions, ground_truth)
            
            # Add physics-based regularization loss
            physics_loss = self.compute_physics_loss(predictions.unsqueeze(1), obj_types)
            
            # Total loss
            loss = prediction_loss + 0.1 * physics_loss
        
        return loss
    
    def compute_physics_loss(self, predictions, obj_types):
        """
        Compute physics-based regularization loss
        
        Args:
            predictions: Trajectory predictions [batch_size, num_samples, output_seq_len, 2]
            obj_types: Object type indices [batch_size]
            
        Returns:
            physics_loss: Physics-based regularization loss
        """
        batch_size = predictions.shape[0]
        device = predictions.device
        
        total_loss = torch.tensor(0.0, device=device)
        
        for b in range(batch_size):
            obj_type = obj_types[b].item()
            constraints = self.physics_constraints.get(obj_type, self.physics_constraints[9])
            
            max_speed, max_accel, max_angular_vel = constraints
            
            # Process each sample for this batch item
            for s in range(predictions.shape[1]):
                trajectory = predictions[b, s]
                
                # Speed constraint
                if trajectory.shape[0] > 1:
                    velocities = trajectory[1:] - trajectory[:-1]  # [seq_len-1, 2]
                    speeds = torch.norm(velocities, dim=1) / 0.1  # [seq_len-1]
                    
                    # Penalize speeds exceeding constraint
                    speed_violation = F.relu(speeds - max_speed)
                    speed_loss = torch.mean(speed_violation)
                    
                    total_loss = total_loss + speed_loss
                
                # Acceleration constraint
                if trajectory.shape[0] > 2:
                    velocities = (trajectory[1:] - trajectory[:-1]) / 0.1  # [seq_len-1, 2] 
                    accelerations = (velocities[1:] - velocities[:-1]) / 0.1  # [seq_len-2, 2]
                    accel_magnitudes = torch.norm(accelerations, dim=1)  # [seq_len-2]
                    
                    # Penalize accelerations exceeding constraint
                    accel_violation = F.relu(accel_magnitudes - max_accel)
                    accel_loss = torch.mean(accel_violation)
                    
                    total_loss = total_loss + accel_loss
                
                # Angular velocity constraint
                if trajectory.shape[0] > 2:
                    directions = velocities / (torch.norm(velocities, dim=1, keepdim=True) + 1e-6)  # [seq_len-1, 2]
                    
                    # Calculate angles between consecutive directions
                    dot_products = torch.sum(directions[1:] * directions[:-1], dim=1)  # [seq_len-2]
                    dot_products = torch.clamp(dot_products, -1.0, 1.0)
                    angles = torch.acos(dot_products) / 0.1  # [seq_len-2]
                    
                    # Penalize angular velocities exceeding constraint
                    angular_violation = F.relu(angles - max_angular_vel)
                    angular_loss = torch.mean(angular_violation)
                    
                    total_loss = total_loss + angular_loss
        
        # Normalize by batch size
        if batch_size > 0:
            total_loss = total_loss / batch_size
        
        return total_loss