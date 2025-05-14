# physics_constraints.py

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from collections import defaultdict

from data_utils.data_utils import TrajectoryDataset

def extract_physics_constraints(data_path, scale=7.0, percentile=95):
    """
    Extract physics constraints from training data for each object type
    
    Args:
        data_path: Path to training data
        scale: Normalization scale factor
        percentile: Percentile to use for max values (to avoid outliers)
        
    Returns:
        physics_constraints: Dictionary of constraints by object type
    """
    print(f"Loading data from {data_path}...")
    dataset = TrajectoryDataset(data_path, split='train', scale=scale, augment=False)
    dataloader = DataLoader(dataset, batch_size=100, shuffle=False)
    
    # Store metrics by object type
    speeds_by_type = defaultdict(list)
    accels_by_type = defaultdict(list)
    angular_vels_by_type = defaultdict(list)
    
    # Object type mapping for reference
    object_types = [
        'vehicle', 'pedestrian', 'motorcyclist', 'cyclist', 'bus', 
        'static', 'background', 'construction', 'riderless_bicycle', 'unknown'
    ]
    
    print("Analyzing physics constraints...")
    for batch in tqdm(dataloader):
        history = batch['history']  # [batch_size, num_agents, seq_len, feature_dim]
        batch_size = history.shape[0]
        
        # Process each sample in the batch
        for b in range(batch_size):
            # Process each agent
            for a in range(history.shape[1]):
                # Check if this agent has valid data
                positions = history[b, a, :, :2]  # [seq_len, 2]
                
                # Skip if no valid positions (all zeros)
                if torch.sum(torch.abs(positions)) == 0:
                    continue
                
                # Get object type
                obj_type = int(history[b, a, 0, 5].item())
                
                # Handle invalid types
                if obj_type < 0 or obj_type >= len(object_types):
                    obj_type = 9  # unknown
                
                # Calculate valid mask (non-zero positions)
                valid_mask = torch.sum(torch.abs(positions), dim=1) > 0
                
                # Skip if less than 2 valid positions
                if torch.sum(valid_mask) < 2:
                    continue
                
                # Extract valid positions
                valid_positions = positions[valid_mask]
                
                # Calculate velocities
                if len(valid_positions) >= 2:
                    velocities = (valid_positions[1:] - valid_positions[:-1]) / 0.1  # 10Hz
                    speeds = torch.norm(velocities, dim=1)
                    
                    # Store speeds for this object type (in normalized space)
                    speeds_by_type[obj_type].extend(speeds.tolist())
                
                # Calculate accelerations
                if len(valid_positions) >= 3:
                    velocity_changes = velocities[1:] - velocities[:-1]
                    accelerations = velocity_changes / 0.1  # 10Hz
                    accel_magnitudes = torch.norm(accelerations, dim=1)
                    
                    # Store accelerations for this object type
                    accels_by_type[obj_type].extend(accel_magnitudes.tolist())
                
                # Calculate angular velocities
                if len(valid_positions) >= 3:
                    # Normalize velocity vectors
                    velocity_dirs = velocities / (torch.norm(velocities, dim=1, keepdim=True) + 1e-6)
                    
                    # Calculate dot products between consecutive velocity directions
                    dots = torch.sum(velocity_dirs[1:] * velocity_dirs[:-1], dim=1)
                    # Clamp to avoid numerical issues
                    dots = torch.clamp(dots, -0.9999, 0.9999)
                    # Calculate angles in radians
                    angles = torch.acos(dots)
                    # Convert to angular velocity (rad/s)
                    angular_velocities = angles / 0.1  # 10Hz
                    
                    # Store angular velocities for this object type
                    angular_vels_by_type[obj_type].extend(angular_velocities.tolist())
    
    # Process results and extract percentiles
    physics_constraints = {}
    
    print("\nPhysics constraints by object type (normalized values):")
    print("=" * 80)
    print(f"{'Object Type':<15} {'Max Speed':>12} {'Max Accel':>12} {'Max Ang Vel':>12}")
    print("-" * 80)
    
    for obj_type in range(len(object_types)):
        # Get data for this object type
        type_speeds = speeds_by_type.get(obj_type, [0])
        type_accels = accels_by_type.get(obj_type, [0])
        type_ang_vels = angular_vels_by_type.get(obj_type, [0])
        
        # Calculate percentiles
        max_speed = np.percentile(type_speeds, percentile) if type_speeds else 0.01
        max_accel = np.percentile(type_accels, percentile) if type_accels else 0.01
        max_ang_vel = np.percentile(type_ang_vels, percentile) if type_ang_vels else 0.01
        
        # Store constraints
        physics_constraints[obj_type] = [
            float(max_speed),
            float(max_accel),
            float(max_ang_vel)
        ]
        
        # Print results
        print(f"{object_types[obj_type]:<15} {max_speed:>12.4f} {max_accel:>12.4f} {max_ang_vel:>12.4f}")
    
    print("=" * 80)
    print(f"Note: Values are in normalized space (scale factor: {scale})")
    print(f"      Real-world values: multiply by {scale} for distances/speeds/accels")
    
    # Generate code snippet for the constraints
    print("\nPhysics constraints code snippet:")
    print("self.physics_constraints = {")
    for obj_type in range(len(object_types)):
        constraints = physics_constraints[obj_type]
        print(f"    {obj_type}: [{constraints[0]:.4f}, {constraints[1]:.4f}, {constraints[2]:.4f}],  # {object_types[obj_type]}")
    print("}")
    
    # Create visualization
    plt.figure(figsize=(15, 5))
    
    # Speed comparison
    plt.subplot(1, 3, 1)
    obj_names = []
    obj_speeds = []
    
    for obj_type in range(len(object_types)):
        if obj_type in speeds_by_type and len(speeds_by_type[obj_type]) > 10:
            obj_names.append(object_types[obj_type])
            obj_speeds.append(physics_constraints[obj_type][0])
    
    plt.bar(obj_names, obj_speeds)
    plt.title(f"{percentile}th Percentile Speed (Normalized)")
    plt.ylabel("Speed")
    plt.xticks(rotation=45, ha='right')
    
    # Acceleration comparison
    plt.subplot(1, 3, 2)
    obj_names = []
    obj_accels = []
    
    for obj_type in range(len(object_types)):
        if obj_type in accels_by_type and len(accels_by_type[obj_type]) > 10:
            obj_names.append(object_types[obj_type])
            obj_accels.append(physics_constraints[obj_type][1])
    
    plt.bar(obj_names, obj_accels)
    plt.title(f"{percentile}th Percentile Acceleration (Normalized)")
    plt.ylabel("Acceleration")
    plt.xticks(rotation=45, ha='right')
    
    # Angular velocity comparison
    plt.subplot(1, 3, 3)
    obj_names = []
    obj_ang_vels = []
    
    for obj_type in range(len(object_types)):
        if obj_type in angular_vels_by_type and len(angular_vels_by_type[obj_type]) > 10:
            obj_names.append(object_types[obj_type])
            obj_ang_vels.append(physics_constraints[obj_type][2])
    
    plt.bar(obj_names, obj_ang_vels)
    plt.title(f"{percentile}th Percentile Angular Velocity (rad/s)")
    plt.ylabel("Angular Velocity")
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig("physics_constraints.png")
    print("Visualization saved to physics_constraints.png")
    
    return physics_constraints

if __name__ == "__main__":
    # Set data path based on environment
    if os.getcwd().startswith('/tscc'):
        train_path = '/tscc/nfs/home/bax001/scratch/CSE_251B/data/train.npz'
    else:
        train_path = 'data/train.npz'
    
    # Extract physics constraints
    constraints = extract_physics_constraints(
        data_path=train_path,
        scale=7.0,  # Use the same scale as in your model
        percentile=95  # Use 95th percentile to avoid outliers
    )

    print(constraints)