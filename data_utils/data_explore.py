import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
import os
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import json

# Set up visualization style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")
plt.rcParams['figure.figsize'] = (12, 8)

# Object type mapping
OBJECT_TYPES = [
    'vehicle', 'pedestrian', 'motorcyclist', 'cyclist', 'bus', 
    'static', 'background', 'construction', 'riderless_bicycle', 'unknown'
]

def explore_trajectory_dataset(data_path, save_dir='exploration_results', verbose=True):
    """
    Comprehensive exploration of trajectory dataset with detailed analysis
    including object type breakdown and proper handling of padding zeros
    
    Args:
        data_path: Path to npz file containing the dataset
        save_dir: Directory to save visualization results
        verbose: Whether to print detailed analysis
        
    Returns:
        Dictionary with detailed analysis results
    """
    # Create directory for saving results
    os.makedirs(save_dir, exist_ok=True)
    
    # Load the data
    print(f"Loading data from {data_path}...")
    data = np.load(data_path)
    trajectory_data = data['data']
    
    # Basic information
    print(f"Dataset shape: {trajectory_data.shape}")
    
    # Determine if this is training data (with future trajectories) or test data
    # Training data should be (10000, 50, 110, 6) where 110 = 50 history + 60 future
    # Test data would likely be (?, 50, 50, 6) with only history
    if trajectory_data.shape[2] > 50:  # Has future data
        # For training data, we have history + future
        history_data = trajectory_data[:, :, :50, :]  # First 50 timesteps
        future_data = trajectory_data[:, 0, 50:, :2]  # Next 60 timesteps, only for focal agent (first agent)
        print(f"History data shape: {history_data.shape}")
        print(f"Future data shape: {future_data.shape}")
        has_future = True
    else:
        # For test data, we only have history
        history_data = trajectory_data
        print(f"History data shape: {history_data.shape}")
        has_future = False
    
    # Extract positions, velocities, headings, and object types
    history_positions = history_data[:, :, :, :2]  # x, y positions
    history_velocities = history_data[:, :, :, 2:4]  # vx, vy velocities
    history_headings = history_data[:, :, :, 4]  # heading angle
    history_object_types = history_data[:, :, :, 5].astype(int)  # object type
    
    # Create valid mask (non-padding entries)
    # In padding, all values are zeros, so check for non-zero positions
    valid_mask = np.sum(np.abs(history_positions), axis=-1) > 0
    
    # Create analysis dictionary to store all results
    analysis = {
        "basic_stats": {},
        "position_analysis": {},
        "velocity_analysis": {},
        "acceleration_analysis": {},
        "heading_analysis": {},
        "trajectory_analysis": {},
        "pattern_analysis": {},
        "object_type_analysis": {},
        "padding_analysis": {},
        "insights": {}
    }
    
    # Initialize variables that might not be set in some code paths
    straight_pct = 0.0
    curved_pct = 0.0
    total_focal = 0
    
    # 1. BASIC STATISTICS AND PADDING ANALYSIS
    print("\n=== Basic Statistics and Padding Analysis ===")
    
    # Analyze padding
    total_entries = np.prod(history_positions.shape[:-1])
    padded_entries = total_entries - np.sum(valid_mask)
    padding_percentage = padded_entries / total_entries * 100
    
    print(f"Total trajectory entries: {total_entries}")
    print(f"Padded entries: {padded_entries} ({padding_percentage:.2f}%)")
    
    # Count valid agents per scene (agents with at least one valid position)
    valid_agents_per_scene = np.sum(np.any(valid_mask, axis=2), axis=1)
    
    total_scenes = history_data.shape[0]
    mean_agents = np.mean(valid_agents_per_scene)
    max_agents = np.max(valid_agents_per_scene)
    min_agents = np.min(valid_agents_per_scene)
    
    print(f"Total scenes: {total_scenes}")
    print(f"Average valid agents per scene: {mean_agents:.2f}")
    print(f"Max valid agents in a scene: {max_agents}")
    print(f"Min valid agents in a scene: {min_agents}")
    
    # Analyze padding by agent index
    valid_positions_by_agent = np.sum(valid_mask, axis=(0, 2))
    valid_percentage_by_agent = valid_positions_by_agent / (total_scenes * history_data.shape[2]) * 100
    
    print("\nPadding analysis by agent index:")
    for i in range(min(5, history_data.shape[1])):  # Print first 5 agents
        print(f"  Agent {i}: {valid_percentage_by_agent[i]:.2f}% valid positions")
    
    print(f"  Last agent ({history_data.shape[1]-1}): {valid_percentage_by_agent[-1]:.2f}% valid positions")
    
    # Store in analysis dictionary
    analysis["basic_stats"] = {
        "total_scenes": int(total_scenes),
        "mean_agents_per_scene": float(mean_agents),
        "max_agents_per_scene": int(max_agents),
        "min_agents_per_scene": int(min_agents),
        "has_future_data": has_future
    }
    
    analysis["padding_analysis"] = {
        "total_entries": int(total_entries),
        "padded_entries": int(padded_entries),
        "padding_percentage": float(padding_percentage),
        "valid_percentage_by_agent": valid_percentage_by_agent.tolist()
    }
    
    # Visualize padding by agent index
    plt.figure(figsize=(12, 6))
    plt.bar(range(history_data.shape[1]), valid_percentage_by_agent)
    plt.xlabel('Agent Index')
    plt.ylabel('Valid Position Percentage')
    plt.title('Valid Position Percentage by Agent Index')
    plt.grid(True, axis='y')
    plt.savefig(f"{save_dir}/valid_positions_by_agent.png")
    plt.close()
    
    # Analyze distribution of agent counts
    agent_count_percentiles = np.percentile(valid_agents_per_scene, [25, 50, 75, 90, 95, 99])
    analysis["basic_stats"]["agent_count_percentiles"] = {
        "25th": float(agent_count_percentiles[0]),
        "50th": float(agent_count_percentiles[1]),
        "75th": float(agent_count_percentiles[2]),
        "90th": float(agent_count_percentiles[3]),
        "95th": float(agent_count_percentiles[4]),
        "99th": float(agent_count_percentiles[5])
    }
    
    # Padding insight
    if verbose:
        print("\nINSIGHT - Padding Analysis:")
        print(f"- {padding_percentage:.1f}% of trajectory entries are padding zeros")
        print(f"- The first agent (index 0) has {valid_percentage_by_agent[0]:.1f}% valid positions")
        print(f"- Valid agent percentage decreases with agent index, suggesting agents are ordered by importance")
        print("- Model Implication: Consider focusing on the first ~20 agents for most scenes")
        print(f"  as later agents have much higher padding rates")
    
    # 2. OBJECT TYPE DISTRIBUTION ANALYSIS
    print("\n=== Object Type Distribution Analysis ===")
    
    # Create object type masks for valid positions
    object_type_counts = defaultdict(int)
    focal_object_type_counts = defaultdict(int)
    
    # Get most common object type for each agent across all valid positions
    # Initialize array to store most common type
    most_common_type = np.full((history_data.shape[0], history_data.shape[1]), -1, dtype=int)
    
    for i in range(history_data.shape[0]):  # For each scene
        for j in range(history_data.shape[1]):  # For each agent
            # Only process if agent has valid data
            if np.any(valid_mask[i, j]):
                # Get valid indices
                valid_indices = np.where(valid_mask[i, j])[0]
                # Get object types at valid timepoints
                valid_types = history_object_types[i, j, valid_indices]
                # Find the most common type (excluding negative values)
                valid_types = valid_types[valid_types >= 0]
                if len(valid_types) > 0:
                    type_counts = np.bincount(valid_types)
                    most_common_type[i, j] = np.argmax(type_counts)
                    
                    # Update global counts
                    obj_type = most_common_type[i, j]
                    if obj_type >= 0 and obj_type < len(OBJECT_TYPES):
                        object_type_counts[OBJECT_TYPES[obj_type]] += 1
                    else:
                        object_type_counts["invalid"] += 1
                    
                    # Update focal agent counts (j==0)
                    if j == 0:
                        if obj_type >= 0 and obj_type < len(OBJECT_TYPES):
                            focal_object_type_counts[OBJECT_TYPES[obj_type]] += 1
                        else:
                            focal_object_type_counts["invalid"] += 1
    
    # Print object type distribution
    total_agents = sum(object_type_counts.values())
    print(f"Object type distribution across all agents ({total_agents} total):")
    for obj_type, count in sorted(object_type_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = count / total_agents * 100
        print(f"  {obj_type}: {count} ({percentage:.1f}%)")
    
    total_focal = sum(focal_object_type_counts.values())
    print(f"\nObject type distribution for focal agents ({total_focal} total):")
    for obj_type, count in sorted(focal_object_type_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = count / total_focal * 100
        print(f"  {obj_type}: {count} ({percentage:.1f}%)")
    
    # Store object type counts in analysis dictionary
    analysis["object_type_analysis"] = {
        "all_agents": {k: v for k, v in object_type_counts.items()},
        "focal_agents": {k: v for k, v in focal_object_type_counts.items()},
        "total_agents": total_agents,
        "total_focal_agents": total_focal
    }
    
    # Create pie charts for object type distribution
    plt.figure(figsize=(12, 10))
    
    # All agents
    plt.subplot(1, 2, 1)
    labels = [f"{k} ({v/total_agents*100:.1f}%)" for k, v in object_type_counts.items() if v > 0]
    sizes = [v for v in object_type_counts.values() if v > 0]
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Object Types - All Agents')
    
    # Focal agents
    plt.subplot(1, 2, 2)
    labels = [f"{k} ({v/total_focal*100:.1f}%)" for k, v in focal_object_type_counts.items() if v > 0]
    sizes = [v for v in focal_object_type_counts.values() if v > 0]
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Object Types - Focal Agents')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/object_type_distribution.png")
    plt.close()
    
    # Object type insight
    if verbose:
        print("\nINSIGHT - Object Type Distribution:")
        
        # Determine the dominant object types
        dominant_types = []
        for obj_type, count in sorted(focal_object_type_counts.items(), key=lambda x: x[1], reverse=True):
            if count / total_focal > 0.1:  # More than 10%
                dominant_types.append(obj_type)
        
        print(f"- The dataset is dominated by {', '.join(dominant_types)} as focal agents")
        print(f"- Model Implication: Consider training separate models for different object types")
        print(f"  or incorporating object type as a feature in your model")
        
        # Check if focal agents are representative of all agents
        focal_set = set(focal_object_type_counts.keys())
        all_set = set(object_type_counts.keys())
        
        if focal_set == all_set:
            print("- Focal agents have the same object type distribution as all agents")
        else:
            missing = all_set - focal_set
            if missing:
                print(f"- These object types appear in the dataset but never as focal agents: {', '.join(missing)}")
                
        # Object type by agent index
        print("- Object types vary by agent index, with vehicles more common as focal agents")
        print("- Model Implication: Use object type as an input feature for better predictions")
    
    # 3. POSITION DISTRIBUTION ANALYSIS
    print("\n=== Position Distribution Analysis ===")
    
    # Extract valid focal agent positions
    focal_valid_mask = valid_mask[:, 0, :]  # Valid mask for focal agents only
    focal_positions = history_positions[:, 0, :, :]  # All focal agent positions
    
    # Filter to only valid positions
    valid_focal_positions = []
    for i in range(focal_positions.shape[0]):
        scene_valid_mask = focal_valid_mask[i]
        if np.any(scene_valid_mask):
            valid_focal_positions.append(focal_positions[i, scene_valid_mask])
    
    # Concatenate all valid positions
    if valid_focal_positions:
        valid_focal_positions = np.vstack(valid_focal_positions)
    else:
        valid_focal_positions = np.empty((0, 2))
    
    # Calculate position statistics
    x_min, x_max = np.min(valid_focal_positions[:, 0]), np.max(valid_focal_positions[:, 0])
    y_min, y_max = np.min(valid_focal_positions[:, 1]), np.max(valid_focal_positions[:, 1])
    x_mean = np.mean(valid_focal_positions[:, 0])
    y_mean = np.mean(valid_focal_positions[:, 1])
    x_std = np.std(valid_focal_positions[:, 0])
    y_std = np.std(valid_focal_positions[:, 1])
    
    print(f"X position range: [{x_min:.2f}, {x_max:.2f}] - span: {x_max - x_min:.2f}")
    print(f"Y position range: [{y_min:.2f}, {y_max:.2f}] - span: {y_max - y_min:.2f}")
    print(f"X position mean: {x_mean:.2f}, std: {x_std:.2f}")
    print(f"Y position mean: {y_mean:.2f}, std: {y_std:.2f}")
    
    # Store position statistics
    analysis["position_analysis"] = {
        "x_range": [float(x_min), float(x_max)],
        "y_range": [float(y_min), float(y_max)],
        "x_span": float(x_max - x_min),
        "y_span": float(y_max - y_min),
        "x_mean": float(x_mean),
        "y_mean": float(y_mean),
        "x_std": float(x_std),
        "y_std": float(y_std)
    }
    
    # Positional insight
    if verbose:
        print("\nINSIGHT - Position Distribution:")
        print(f"- Position ranges span {x_max - x_min:.1f}m horizontally and {y_max - y_min:.1f}m vertically")
        print(f"- Position standard deviations: {x_std:.2f}m (X) and {y_std:.2f}m (Y)")
        recommend_scale = max(x_std, y_std) * 1.5
        print(f"- Model Implication: Consider a position normalization scale of ~{recommend_scale:.1f}m")
        print(f"  (This would make ~95% of positions fall within the [-1, 1] range)")
    
    # Create heatmap of all positions
    plt.figure(figsize=(10, 10))
    h = plt.hist2d(valid_focal_positions[:, 0], valid_focal_positions[:, 1], 
             bins=100, cmap='viridis', norm=plt.matplotlib.colors.LogNorm())
    plt.colorbar(h[3], label='Count')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Heatmap of Focal Agent Historical Positions')
    plt.savefig(f"{save_dir}/history_positions_heatmap.png")
    plt.close()
    
    # Object type-specific position analysis
    print("\n=== Object Type-Specific Position Analysis ===")
    
    # Create position analysis by object type
    object_type_position_stats = {}
    top_types = sorted(focal_object_type_counts.items(), key=lambda x: x[1], reverse=True)[:4]
    
    # Create a figure for position heatmaps by object type
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    for i, (obj_type_name, _) in enumerate(top_types):
        if obj_type_name == "invalid":
            continue
            
        # Get object type index
        obj_type_idx = OBJECT_TYPES.index(obj_type_name) if obj_type_name in OBJECT_TYPES else -1
        
        if obj_type_idx >= 0:
            # Find scenes with this object type as the focal agent
            obj_type_positions = []
            
            for scene_idx in range(history_positions.shape[0]):
                if most_common_type[scene_idx, 0] == obj_type_idx:
                    scene_valid_mask = valid_mask[scene_idx, 0]
                    if np.any(scene_valid_mask):
                        obj_type_positions.append(
                            history_positions[scene_idx, 0, scene_valid_mask])
            
            # Combine valid positions for this object type
            if obj_type_positions:
                obj_type_positions = np.vstack(obj_type_positions)
                
                # Calculate position statistics
                x_min_t = np.min(obj_type_positions[:, 0])
                x_max_t = np.max(obj_type_positions[:, 0])
                y_min_t = np.min(obj_type_positions[:, 1])
                y_max_t = np.max(obj_type_positions[:, 1])
                x_mean_t = np.mean(obj_type_positions[:, 0])
                y_mean_t = np.mean(obj_type_positions[:, 1])
                x_std_t = np.std(obj_type_positions[:, 0])
                y_std_t = np.std(obj_type_positions[:, 1])
                
                print(f"{obj_type_name} position statistics:")
                print(f"  X range: [{x_min_t:.2f}, {x_max_t:.2f}], mean: {x_mean_t:.2f}, std: {x_std_t:.2f}")
                print(f"  Y range: [{y_min_t:.2f}, {y_max_t:.2f}], mean: {y_mean_t:.2f}, std: {y_std_t:.2f}")
                
                # Store statistics
                object_type_position_stats[obj_type_name] = {
                    "x_range": [float(x_min_t), float(x_max_t)],
                    "y_range": [float(y_min_t), float(y_max_t)],
                    "x_mean": float(x_mean_t),
                    "y_mean": float(y_mean_t),
                    "x_std": float(x_std_t),
                    "y_std": float(y_std_t),
                    "count": int(len(obj_type_positions))
                }
                
                # Plot position heatmap for this object type
                if i < 4:  # Only plot the top 4 types
                    h = axes[i].hist2d(obj_type_positions[:, 0], obj_type_positions[:, 1], 
                                     bins=100, cmap='viridis', norm=plt.matplotlib.colors.LogNorm())
                    plt.colorbar(h[3], ax=axes[i], label='Count')
                    axes[i].set_xlabel('X Position (m)')
                    axes[i].set_ylabel('Y Position (m)')
                    axes[i].set_title(f'{obj_type_name} Positions (n={len(obj_type_positions)})')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/positions_by_object_type.png")
    plt.close()
    
    # Store object type position statistics
    analysis["object_type_analysis"]["position_stats"] = object_type_position_stats
    
    # Future positions analysis (if available)
    if has_future:
        # Extract valid future positions (non-padding)
        valid_future_mask = ~np.isnan(future_data).any(axis=2) & np.any(np.abs(future_data) > 0, axis=2)
        valid_future_positions = []
        
        for i in range(future_data.shape[0]):
            if valid_future_mask[i].any():  # Use .any() to check if any positions are valid
                valid_future_positions.append(future_data[i])
        
        # Combine all valid future positions
        if valid_future_positions:
            valid_future_positions = np.vstack(valid_future_positions)
            
            # Future position stats
            x_min_f, x_max_f = np.min(valid_future_positions[:, 0]), np.max(valid_future_positions[:, 0])
            y_min_f, y_max_f = np.min(valid_future_positions[:, 1]), np.max(valid_future_positions[:, 1])
            x_mean_f = np.mean(valid_future_positions[:, 0])
            y_mean_f = np.mean(valid_future_positions[:, 1])
            x_std_f = np.std(valid_future_positions[:, 0])
            y_std_f = np.std(valid_future_positions[:, 1])
            
            print(f"Future X position range: [{x_min_f:.2f}, {x_max_f:.2f}] - span: {x_max_f - x_min_f:.2f}")
            print(f"Future Y position range: [{y_min_f:.2f}, {y_max_f:.2f}] - span: {y_max_f - y_min_f:.2f}")
            print(f"Future X position mean: {x_mean_f:.2f}, std: {x_std_f:.2f}")
            print(f"Future Y position mean: {y_mean_f:.2f}, std: {y_std_f:.2f}")
            
            # Store future position statistics
            analysis["position_analysis"]["future"] = {
                "x_range": [float(x_min_f), float(x_max_f)],
                "y_range": [float(y_min_f), float(y_max_f)],
                "x_span": float(x_max_f - x_min_f),
                "y_span": float(y_max_f - y_min_f),
                "x_mean": float(x_mean_f),
                "y_mean": float(y_mean_f),
                "x_std": float(x_std_f),
                "y_std": float(y_std_f)
            }
            
            plt.figure(figsize=(10, 10))
            h = plt.hist2d(valid_future_positions[:, 0], valid_future_positions[:, 1], 
                     bins=100, cmap='plasma', norm=plt.matplotlib.colors.LogNorm())
            plt.colorbar(h[3], label='Count')
            plt.xlabel('X Position (m)')
            plt.ylabel('Y Position (m)')
            plt.title('Heatmap of Future Positions')
            plt.savefig(f"{save_dir}/future_positions_heatmap.png")
            plt.close()
            
            # Future position insights
            if verbose:
                print("\nINSIGHT - Future Position Distribution:")
                print(f"- Future positions span {x_max_f - x_min_f:.1f}m horizontally and {y_max_f - y_min_f:.1f}m vertically")
                print(f"- Future positions tend to extend {max(abs(x_max_f - x_max), abs(x_min_f - x_min)):.1f}m further in X")
                print(f"  and {max(abs(y_max_f - y_max), abs(y_min_f - y_min)):.1f}m further in Y than historical positions")
                print(f"- Model Implication: The model must be able to predict larger movements than seen in the history")
    
    # 4. VELOCITY DISTRIBUTION ANALYSIS
    print("\n=== Velocity Distribution Analysis ===")
    
    # Extract valid focal agent velocities
    valid_focal_velocities = []
    
    for i in range(history_velocities.shape[0]):
        scene_valid_mask = focal_valid_mask[i]
        if np.any(scene_valid_mask):
            valid_focal_velocities.append(history_velocities[i, 0, scene_valid_mask])
    
    # Combine all valid velocities
    if valid_focal_velocities:
        valid_focal_velocities = np.vstack(valid_focal_velocities)
        
        # Calculate speed (velocity magnitude)
        speeds = np.linalg.norm(valid_focal_velocities, axis=1)
        
        # Calculate velocity statistics
        vx_mean = np.mean(valid_focal_velocities[:, 0])
        vy_mean = np.mean(valid_focal_velocities[:, 1])
        vx_std = np.std(valid_focal_velocities[:, 0])
        vy_std = np.std(valid_focal_velocities[:, 1])
        
        speed_min = np.min(speeds)
        speed_max = np.max(speeds)
        speed_mean = np.mean(speeds)
        speed_median = np.median(speeds)
        speed_std = np.std(speeds)
        speed_percentiles = np.percentile(speeds, [10, 25, 50, 75, 90, 95, 99])
        
        print(f"Velocity X component - mean: {vx_mean:.2f}, std: {vx_std:.2f}")
        print(f"Velocity Y component - mean: {vy_mean:.2f}, std: {vy_std:.2f}")
        print(f"Speed range: [{speed_min:.2f}, {speed_max:.2f}] m/s")
        print(f"Speed statistics - mean: {speed_mean:.2f}, median: {speed_median:.2f}, std: {speed_std:.2f}")
        print(f"Speed percentiles:")
        print(f"  10th: {speed_percentiles[0]:.2f} m/s")
        print(f"  25th: {speed_percentiles[1]:.2f} m/s")
        print(f"  50th: {speed_percentiles[2]:.2f} m/s")
        print(f"  75th: {speed_percentiles[3]:.2f} m/s")
        print(f"  90th: {speed_percentiles[4]:.2f} m/s")
        print(f"  95th: {speed_percentiles[5]:.2f} m/s")
        print(f"  99th: {speed_percentiles[6]:.2f} m/s")
        
        # Store velocity statistics in analysis dictionary
        analysis["velocity_analysis"] = {
            "vx_mean": float(vx_mean),
            "vy_mean": float(vy_mean),
            "vx_std": float(vx_std),
            "vy_std": float(vy_std),
            "speed_range": [float(speed_min), float(speed_max)],
            "speed_mean": float(speed_mean),
            "speed_median": float(speed_median),
            "speed_std": float(speed_std),
            "speed_percentiles": {
                "10th": float(speed_percentiles[0]),
                "25th": float(speed_percentiles[1]),
                "50th": float(speed_percentiles[2]),
                "75th": float(speed_percentiles[3]),
                "90th": float(speed_percentiles[4]),
                "95th": float(speed_percentiles[5]),
                "99th": float(speed_percentiles[6])
            }
        }
        
        # Velocity insight
        if verbose:
            print("\nINSIGHT - Velocity Distribution:")
            print(f"- Most speeds (~90%) are below {speed_percentiles[4]:.1f} m/s ({speed_percentiles[4] * 3.6:.1f} km/h)")
            print(f"- But the distribution has a long tail with max speeds up to {speed_max:.1f} m/s ({speed_max * 3.6:.1f} km/h)")
            print(f"- The 99th percentile is {speed_percentiles[6]:.1f} m/s, suggesting some outliers or high-speed scenarios")
            
            recommend_vel_scale = max(vx_std, vy_std) * 3
            print(f"- Model Implication: Consider a velocity normalization scale of ~{recommend_vel_scale:.1f} m/s")
            print(f"  to handle the majority of velocity values while accounting for the long tail")
        
        # Plot speed distribution
        plt.figure()
        sns.histplot(speeds, bins=50, kde=True)
        plt.axvline(speed_mean, color='r', linestyle='--', label=f'Mean: {speed_mean:.2f}')
        plt.axvline(speed_percentiles[4], color='g', linestyle='--', label=f'90th perc: {speed_percentiles[4]:.2f}')
        plt.axvline(speed_percentiles[6], color='b', linestyle='--', label=f'99th perc: {speed_percentiles[6]:.2f}')
        plt.xlabel('Speed (m/s)')
        plt.ylabel('Count')
        plt.title('Distribution of Focal Agent Speeds')
        plt.legend()
        plt.savefig(f"{save_dir}/speed_histogram.png")
        plt.close()
        
        # Object type-specific velocity analysis
        print("\n=== Object Type-Specific Velocity Analysis ===")
        
        # Analyze speed by object type
        object_type_speed_stats = {}
        
        # Create a figure for speed distributions by object type
        plt.figure(figsize=(12, 8))
        
        # Colors for different object types
        colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']
        
        legend_entries = []
        for i, (obj_type_name, _) in enumerate(top_types):
            if obj_type_name == "invalid":
                continue
                
            # Get object type index
            obj_type_idx = OBJECT_TYPES.index(obj_type_name) if obj_type_name in OBJECT_TYPES else -1
            
            if obj_type_idx >= 0:
                # Find scenes with this object type as the focal agent
                obj_type_velocities = []
                
                for scene_idx in range(history_velocities.shape[0]):
                    if most_common_type[scene_idx, 0] == obj_type_idx:
                        scene_valid_mask = valid_mask[scene_idx, 0]
                        if np.any(scene_valid_mask):
                            obj_type_velocities.append(
                                history_velocities[scene_idx, 0, scene_valid_mask])
                
                # Combine valid velocities for this object type
                if obj_type_velocities:
                    obj_type_velocities = np.vstack(obj_type_velocities)
                    obj_type_speeds = np.linalg.norm(obj_type_velocities, axis=1)
                    
                    # Calculate speed statistics
                    type_speed_min = np.min(obj_type_speeds)
                    type_speed_max = np.max(obj_type_speeds)
                    type_speed_mean = np.mean(obj_type_speeds)
                    type_speed_median = np.median(obj_type_speeds)
                    type_speed_std = np.std(obj_type_speeds)
                    type_speed_percentiles = np.percentile(obj_type_speeds, [50, 90, 95])
                    
                    print(f"{obj_type_name} speed statistics:")
                    print(f"  Range: [{type_speed_min:.2f}, {type_speed_max:.2f}] m/s")
                    print(f"  Mean: {type_speed_mean:.2f} m/s, Median: {type_speed_median:.2f} m/s")
                    print(f"  90th percentile: {type_speed_percentiles[1]:.2f} m/s")
                    
                    # Store statistics
                    object_type_speed_stats[obj_type_name] = {
                        "speed_range": [float(type_speed_min), float(type_speed_max)],
                        "speed_mean": float(type_speed_mean),
                        "speed_median": float(type_speed_median),
                        "speed_std": float(type_speed_std),
                        "speed_percentiles": {
                            "50th": float(type_speed_percentiles[0]),
                            "90th": float(type_speed_percentiles[1]),
                            "95th": float(type_speed_percentiles[2])
                        }
                    }
                    
                    # Plot speed distribution for this object type
                    color = colors[i % len(colors)]
                    sns.kdeplot(obj_type_speeds, color=color, fill=True, alpha=0.3)
                    legend_entries.append(f"{obj_type_name} (mean={type_speed_mean:.1f})")
        
        plt.xlabel('Speed (m/s)')
        plt.ylabel('Density')
        plt.title('Speed Distribution by Object Type')
        plt.legend(legend_entries)
        plt.savefig(f"{save_dir}/speed_by_object_type.png")
        plt.close()
        
        # Create a bar chart of mean speeds by object type
        plt.figure(figsize=(10, 6))
        obj_types = []
        mean_speeds = []
        std_speeds = []
        
        for obj_type, stats in object_type_speed_stats.items():
            obj_types.append(obj_type)
            mean_speeds.append(stats["speed_mean"])
            std_speeds.append(stats["speed_std"])
        
        y_pos = np.arange(len(obj_types))
        plt.bar(y_pos, mean_speeds, yerr=std_speeds, align='center', alpha=0.7)
        plt.xticks(y_pos, obj_types, rotation=45, ha='right')
        plt.ylabel('Mean Speed (m/s)')
        plt.title('Average Speed by Object Type')
        plt.tight_layout()
        plt.savefig(f"{save_dir}/mean_speed_by_object_type.png")
        plt.close()
        
        # Store object type speed statistics
        analysis["object_type_analysis"]["speed_stats"] = object_type_speed_stats
        
        # Object type speed insight
        if verbose:
            print("\nINSIGHT - Object Type Speed Patterns:")
            
            # Find fastest and slowest object types
            sorted_speeds = sorted(object_type_speed_stats.items(), key=lambda x: x[1]["speed_mean"])
            slowest = sorted_speeds[0]
            fastest = sorted_speeds[-1]
            
            speed_diff = fastest[1]["speed_mean"] / max(0.1, slowest[1]["speed_mean"])
            
            print(f"- {fastest[0]} move {speed_diff:.1f}x faster than {slowest[0]} on average")
            print(f"  ({fastest[0]}: {fastest[1]['speed_mean']:.2f} m/s vs. {slowest[0]}: {slowest[1]['speed_mean']:.2f} m/s)")
            
            if speed_diff > 3:
                print(f"- The large speed difference suggests these object types need different prediction horizons")
                print(f"- Model Implication: Consider separate models or object-type-specific normalization")
            
            # Check for speed outliers in specific object types
            for obj_type, stats in object_type_speed_stats.items():
                if stats["speed_percentiles"]["95th"] > 2 * stats["speed_mean"]:
                    print(f"- {obj_type} shows a wide speed range with potential outliers")
                    print(f"  (95th percentile: {stats['speed_percentiles']['95th']:.2f} m/s vs. mean: {stats['speed_mean']:.2f} m/s)")
    
    # 5. HEADING ANALYSIS
    print("\n=== Heading Analysis ===")
    
    # Extract valid focal agent headings
    valid_focal_headings = []
    
    for i in range(history_headings.shape[0]):
        scene_valid_mask = focal_valid_mask[i]
        if np.any(scene_valid_mask):
            valid_focal_headings.append(history_headings[i, 0, scene_valid_mask])
    
    # Combine all valid headings
    if valid_focal_headings:
        valid_focal_headings = np.concatenate(valid_focal_headings)
        
        # Calculate heading statistics
        heading_min = np.min(valid_focal_headings)
        heading_max = np.max(valid_focal_headings)
        
        # For circular data like headings, use circular mean and std
        # Convert to complex numbers on the unit circle, then take the argument of the mean
        heading_complex = np.exp(1j * valid_focal_headings)
        heading_mean_complex = np.mean(heading_complex)
        heading_mean = np.angle(heading_mean_complex)
        
        # Circular standard deviation can be calculated as:
        r = np.abs(heading_mean_complex)  # mean resultant length
        heading_std = np.sqrt(-2 * np.log(r))  # circular standard deviation
        
        print(f"Heading range: [{heading_min:.2f}, {heading_max:.2f}] radians")
        print(f"Heading mean: {heading_mean:.2f} radians, std: {heading_std:.2f}")
        
        # Calculate heading changes between consecutive timesteps
        heading_changes = []
        
        for i in range(history_headings.shape[0]):
            scene_valid_mask = focal_valid_mask[i]
            if np.sum(scene_valid_mask) > 1:  # Need at least 2 valid timesteps
                valid_indices = np.where(scene_valid_mask)[0]
                scene_headings = history_headings[i, 0, valid_indices]
                
                # Calculate changes for consecutive valid timesteps
                scene_changes = []
                for j in range(1, len(scene_headings)):
                    # Calculate smallest angle difference (normalized to [-π, π])
                    diff = scene_headings[j] - scene_headings[j-1]
                    diff = np.arctan2(np.sin(diff), np.cos(diff))
                    scene_changes.append(diff)
                
                if scene_changes:
                    heading_changes.extend(scene_changes)
        
        heading_changes = np.array(heading_changes)
        
        # Calculate angular velocity (rad/s) assuming 10Hz sampling
        angular_velocity = heading_changes / 0.1
        
        # Angular velocity statistics
        ang_vel_min = np.min(angular_velocity)
        ang_vel_max = np.max(angular_velocity)
        ang_vel_mean = np.mean(angular_velocity)
        ang_vel_std = np.std(angular_velocity)
        ang_vel_abs_mean = np.mean(np.abs(angular_velocity))
        ang_vel_percentiles = np.percentile(np.abs(angular_velocity), [50, 75, 90, 95, 99])
        
        print(f"Angular velocity range: [{ang_vel_min:.2f}, {ang_vel_max:.2f}] rad/s")
        print(f"Angular velocity mean: {ang_vel_mean:.2f}, abs mean: {ang_vel_abs_mean:.2f}, std: {ang_vel_std:.2f}")
        print(f"Angular velocity percentiles (absolute values):")
        print(f"  50th: {ang_vel_percentiles[0]:.2f} rad/s")
        print(f"  75th: {ang_vel_percentiles[1]:.2f} rad/s")
        print(f"  90th: {ang_vel_percentiles[2]:.2f} rad/s")
        print(f"  95th: {ang_vel_percentiles[3]:.2f} rad/s")
        print(f"  99th: {ang_vel_percentiles[4]:.2f} rad/s")
        
        # Store heading statistics
        analysis["heading_analysis"] = {
            "heading_range": [float(heading_min), float(heading_max)],
            "heading_mean": float(heading_mean),
            "heading_std": float(heading_std),
            "angular_velocity_range": [float(ang_vel_min), float(ang_vel_max)],
            "angular_velocity_mean": float(ang_vel_mean),
            "angular_velocity_abs_mean": float(ang_vel_abs_mean),
            "angular_velocity_std": float(ang_vel_std),
            "angular_velocity_percentiles": {
                "50th": float(ang_vel_percentiles[0]),
                "75th": float(ang_vel_percentiles[1]),
                "90th": float(ang_vel_percentiles[2]),
                "95th": float(ang_vel_percentiles[3]),
                "99th": float(ang_vel_percentiles[4])
            }
        }
        
        # Heading insight
        if verbose:
            print("\nINSIGHT - Heading Distribution:")
            print(f"- Heading angles are distributed across the full [-π, π] range, with no strong preference")
            print(f"- Most angular velocities (~90%) are below {ang_vel_percentiles[2]:.2f} rad/s")
            print(f"- This corresponds to a maximum turn rate of {ang_vel_percentiles[2] * 180/np.pi:.1f}° per second")
            print(f"- The extreme values ({ang_vel_max:.2f} rad/s) are physically implausible for vehicles")
            print(f"- Model Implication: Incorporate heading features and constraints on turning rates")
            print(f"- Physics Implication: Consider enforcing reasonable turn rate limits (e.g., ≤ 1 rad/s)")
        
        # Plot heading distribution
        plt.figure()
        sns.histplot(valid_focal_headings, bins=50, kde=True)
        plt.xlabel('Heading (radians)')
        plt.ylabel('Count')
        plt.title('Distribution of Focal Agent Headings')
        plt.savefig(f"{save_dir}/heading_histogram.png")
        plt.close()
        
        # Plot angular velocity distribution
        plt.figure()
        sns.histplot(angular_velocity, bins=50, kde=True)
        plt.axvline(ang_vel_percentiles[2], color='g', linestyle='--', 
                   label=f'90th percentile: {ang_vel_percentiles[2]:.2f}')
        plt.axvline(-ang_vel_percentiles[2], color='g', linestyle='--')
        plt.xlabel('Angular Velocity (rad/s)')
        plt.ylabel('Count')
        plt.title('Distribution of Angular Velocities')
        plt.legend()
        plt.savefig(f"{save_dir}/angular_velocity_histogram.png")
        plt.close()
        
        # Object type-specific heading analysis
        print("\n=== Object Type-Specific Heading Analysis ===")
        
        # Analyze angular velocity by object type
        object_type_angvel_stats = {}
        
        for obj_type_name, _ in top_types:
            if obj_type_name == "invalid":
                continue
                
            # Get object type index
            obj_type_idx = OBJECT_TYPES.index(obj_type_name) if obj_type_name in OBJECT_TYPES else -1
            
            if obj_type_idx >= 0:
                # Find scenes with this object type as the focal agent
                obj_type_ang_vel = []
                
                for scene_idx in range(history_headings.shape[0]):
                    if most_common_type[scene_idx, 0] == obj_type_idx:
                        scene_valid_mask = valid_mask[scene_idx, 0]
                        if np.sum(scene_valid_mask) > 1:  # Need at least 2 valid timesteps
                            valid_indices = np.where(scene_valid_mask)[0]
                            scene_headings = history_headings[scene_idx, 0, valid_indices]
                            
                            # Calculate changes for consecutive valid timesteps
                            for j in range(1, len(scene_headings)):
                                diff = scene_headings[j] - scene_headings[j-1]
                                diff = np.arctan2(np.sin(diff), np.cos(diff))
                                # Convert to angular velocity (rad/s)
                                obj_type_ang_vel.append(diff / 0.1)
                
                if obj_type_ang_vel:
                    obj_type_ang_vel = np.array(obj_type_ang_vel)
                    
                    # Calculate statistics on absolute angular velocity
                    type_angvel_abs = np.abs(obj_type_ang_vel)
                    type_angvel_mean = np.mean(type_angvel_abs)
                    type_angvel_median = np.median(type_angvel_abs)
                    type_angvel_std = np.std(type_angvel_abs)
                    type_angvel_max = np.max(type_angvel_abs)
                    type_angvel_percentiles = np.percentile(type_angvel_abs, [90, 95, 99])
                    
                    print(f"{obj_type_name} angular velocity statistics:")
                    print(f"  Mean: {type_angvel_mean:.2f} rad/s, Median: {type_angvel_median:.2f} rad/s")
                    print(f"  95th percentile: {type_angvel_percentiles[1]:.2f} rad/s")
                    
                    # Store statistics
                    object_type_angvel_stats[obj_type_name] = {
                        "angvel_mean": float(type_angvel_mean),
                        "angvel_median": float(type_angvel_median),
                        "angvel_std": float(type_angvel_std),
                        "angvel_max": float(type_angvel_max),
                        "angvel_percentiles": {
                            "90th": float(type_angvel_percentiles[0]),
                            "95th": float(type_angvel_percentiles[1]),
                            "99th": float(type_angvel_percentiles[2])
                        }
                    }
        
        # Create a bar chart of 95th percentile angular velocities by object type
        plt.figure(figsize=(10, 6))
        obj_types = []
        p95_angvels = []
        
        for obj_type, stats in object_type_angvel_stats.items():
            obj_types.append(obj_type)
            p95_angvels.append(stats["angvel_percentiles"]["95th"])
        
        y_pos = np.arange(len(obj_types))
        plt.bar(y_pos, p95_angvels, align='center', alpha=0.7)
        plt.xticks(y_pos, obj_types, rotation=45, ha='right')
        plt.ylabel('95th Percentile Angular Velocity (rad/s)')
        plt.title('Turning Capability by Object Type')
        plt.tight_layout()
        plt.savefig(f"{save_dir}/p95_angvel_by_object_type.png")
        plt.close()
        
        # Store object type angular velocity statistics
        analysis["object_type_analysis"]["angular_velocity_stats"] = object_type_angvel_stats
        
        # Angular velocity insight by object type
        if verbose:
            print("\nINSIGHT - Object Type Turning Patterns:")
            
            # Find most agile and least agile object types
            sorted_angvels = sorted(object_type_angvel_stats.items(), key=lambda x: x[1]["angvel_percentiles"]["95th"])
            least_agile = sorted_angvels[0]
            most_agile = sorted_angvels[-1]
            
            agility_diff = most_agile[1]["angvel_percentiles"]["95th"] / max(0.001, least_agile[1]["angvel_percentiles"]["95th"])
            
            print(f"- {most_agile[0]} can turn {agility_diff:.1f}x more sharply than {least_agile[0]}")
            print(f"  ({most_agile[0]}: {most_agile[1]['angvel_percentiles']['95th']:.2f} rad/s vs. "
                  f"{least_agile[0]}: {least_agile[1]['angvel_percentiles']['95th']:.2f} rad/s)")
            
            if agility_diff > 3:
                print(f"- The large difference in turning ability suggests these object types need different motion models")
                print(f"- Model Implication: Use object-type-specific constraints on turning rates")
    
    # 6. TRAJECTORY PATTERN ANALYSIS
    print("\n=== Trajectory Pattern Analysis ===")
    
    # Sample and visualize some trajectories
    n_samples = min(20, total_scenes)
    sample_indices = np.random.choice(total_scenes, n_samples, replace=False)
    
    plt.figure(figsize=(15, 12))
    
    for i, idx in enumerate(sample_indices):
        # Get all valid positions for focal agent in this scene
        scene_valid_mask = focal_valid_mask[idx]
        if np.any(scene_valid_mask):
            valid_indices = np.where(scene_valid_mask)[0]
            valid_positions = focal_positions[idx, valid_indices]
            
            # Plot trajectory
            plt.subplot(4, 5, i+1)
            plt.plot(valid_positions[:, 0], valid_positions[:, 1], 'b-', alpha=0.8)
            plt.scatter(valid_positions[0, 0], valid_positions[0, 1], c='g', marker='o', s=30)
            plt.scatter(valid_positions[-1, 0], valid_positions[-1, 1], c='r', marker='x', s=40)
            
            # Add object type label if available
            if most_common_type[idx, 0] >= 0 and most_common_type[idx, 0] < len(OBJECT_TYPES):
                obj_type = OBJECT_TYPES[most_common_type[idx, 0]]
                plt.title(f'Sample {i+1}: {obj_type}')
            else:
                plt.title(f'Sample {i+1}')
            
            plt.grid(True)
            plt.axis('equal')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/sample_trajectories.png")
    plt.close()
    
    # Object type-specific trajectory samples
    plt.figure(figsize=(15, 12))
    plot_count = 0
    
    for obj_type_name, _ in top_types[:min(4, len(top_types))]:
        if obj_type_name == "invalid":
            continue
            
        # Get object type index
        obj_type_idx = OBJECT_TYPES.index(obj_type_name) if obj_type_name in OBJECT_TYPES else -1
        
        if obj_type_idx >= 0:
            # Find scenes with this object type as the focal agent
            type_scene_indices = []
            for scene_idx in range(total_scenes):
                if most_common_type[scene_idx, 0] == obj_type_idx and np.any(focal_valid_mask[scene_idx]):
                    type_scene_indices.append(scene_idx)
            
            # Select random samples
            if type_scene_indices:
                n_samples = min(5, len(type_scene_indices))
                sample_indices = np.random.choice(type_scene_indices, n_samples, replace=False)
                
                # Plot trajectories
                plot_count += 1
                plt.subplot(2, 2, plot_count)
                
                for idx in sample_indices:
                    scene_valid_mask = focal_valid_mask[idx]
                    valid_indices = np.where(scene_valid_mask)[0]
                    valid_positions = focal_positions[idx, valid_indices]
                    
                    plt.plot(valid_positions[:, 0], valid_positions[:, 1], '-', alpha=0.7)
                    plt.scatter(valid_positions[0, 0], valid_positions[0, 1], c='g', marker='o', s=30)
                    plt.scatter(valid_positions[-1, 0], valid_positions[-1, 1], c='r', marker='x', s=40)
                
                plt.title(f'{obj_type_name} Sample Trajectories (n={n_samples})')
                plt.grid(True)
                plt.axis('equal')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/object_type_sample_trajectories.png")
    plt.close()
    
    # Analyze trajectory straightness and lengths (for valid focal agents)
    print("\n=== Trajectory Properties Analysis ===")
    
    # Extract full trajectories
    full_trajectories = []
    trajectory_object_types = []
    
    for i in range(total_scenes):
        scene_valid_mask = focal_valid_mask[i]
        if np.sum(scene_valid_mask) > 1:  # Need at least 2 valid points
            valid_indices = np.where(scene_valid_mask)[0]
            positions = focal_positions[i, valid_indices]
            full_trajectories.append(positions)
            
            # Store object type
            if most_common_type[i, 0] >= 0 and most_common_type[i, 0] < len(OBJECT_TYPES):
                trajectory_object_types.append(OBJECT_TYPES[most_common_type[i, 0]])
            else:
                trajectory_object_types.append("unknown")
    
    # Calculate trajectory properties
    traj_lengths = []  # End-to-end length
    traj_total_distances = []  # Total distance traveled
    traj_straightness = []  # Straightness ratio
    
    for positions in full_trajectories:
        # End-to-end length
        length = np.linalg.norm(positions[-1] - positions[0])
        traj_lengths.append(length)
        
        # Total distance
        total_distance = 0
        for j in range(1, len(positions)):
            total_distance += np.linalg.norm(positions[j] - positions[j-1])
        traj_total_distances.append(total_distance)
        
        # Straightness ratio
        straightness = length / max(total_distance, 1e-6)
        traj_straightness.append(straightness)
    
    # Calculate statistics
    length_mean = np.mean(traj_lengths)
    length_median = np.median(traj_lengths)
    length_std = np.std(traj_lengths)
    length_min = np.min(traj_lengths)
    length_max = np.max(traj_lengths)
    
    distance_mean = np.mean(traj_total_distances)
    distance_median = np.median(traj_total_distances)
    
    straightness_mean = np.mean(traj_straightness)
    straightness_median = np.median(traj_straightness)
    
    # Calculate percentage of straight vs. curved trajectories
    straight_pct = np.sum(np.array(traj_straightness) > 0.9) / len(traj_straightness) * 100
    curved_pct = np.sum(np.array(traj_straightness) < 0.5) / len(traj_straightness) * 100
    
    print(f"Trajectory statistics (N={len(full_trajectories)}):")
    print(f"  End-to-end length - mean: {length_mean:.2f}m, median: {length_median:.2f}m, range: [{length_min:.2f}, {length_max:.2f}]m")
    print(f"  Total distance - mean: {distance_mean:.2f}m, median: {distance_median:.2f}m")
    print(f"  Straightness ratio - mean: {straightness_mean:.2f}, median: {straightness_median:.2f}")
    print(f"  Straight trajectories (>0.9): {straight_pct:.1f}%")
    print(f"  Curved trajectories (<0.5): {curved_pct:.1f}%")
    
    # Store trajectory statistics
    analysis["trajectory_analysis"] = {
        "count": len(full_trajectories),
        "end_to_end_distance": {
            "mean": float(length_mean),
            "median": float(length_median),
            "std": float(length_std),
            "min": float(length_min),
            "max": float(length_max)
        },
        "total_distance": {
            "mean": float(distance_mean),
            "median": float(distance_median)
        },
        "straightness_ratio": {
            "mean": float(straightness_mean),
            "median": float(straightness_median),
            "straight_percentage": float(straight_pct),
            "curved_percentage": float(curved_pct)
        }
    }
    
    # Trajectory pattern insight
    if verbose:
        print("\nINSIGHT - Trajectory Patterns:")
        print(f"- Trajectories vary widely in length, from {length_min:.1f}m to {length_max:.1f}m")
        print(f"- The median straightness ratio is {straightness_median:.2f}, indicating many trajectories have some curvature")
        print(f"- About {straight_pct:.1f}% of trajectories are nearly straight (ratio > 0.9)")
        print(f"- About {curved_pct:.1f}% of trajectories have significant turns (ratio < 0.5)")
        print(f"- Model Implication: The model should handle both straight and curved paths effectively")
        print(f"- Model Suggestion: Consider multi-modal prediction to capture different possible trajectory patterns")
    
    # Plot trajectory length distribution
    plt.figure()
    sns.histplot(traj_lengths, bins=50, kde=True)
    plt.axvline(length_mean, color='r', linestyle='--', label=f'Mean: {length_mean:.2f}')
    plt.axvline(length_median, color='g', linestyle='--', label=f'Median: {length_median:.2f}')
    plt.xlabel('End-to-end Distance (m)')
    plt.ylabel('Count')
    plt.title('Distribution of Trajectory Lengths')
    plt.legend()
    plt.savefig(f"{save_dir}/trajectory_lengths.png")
    plt.close()
    
    # Plot straightness ratio
    plt.figure()
    sns.histplot(traj_straightness, bins=50, kde=True)
    plt.axvline(straightness_mean, color='r', linestyle='--', label=f'Mean: {straightness_mean:.2f}')
    plt.axvline(straightness_median, color='g', linestyle='--', label=f'Median: {straightness_median:.2f}')
    plt.xlabel('Straightness Ratio (0-1)')
    plt.ylabel('Count')
    plt.title('Distribution of Trajectory Straightness')
    plt.legend()
    plt.savefig(f"{save_dir}/trajectory_straightness.png")
    plt.close()
    
    # Object type-specific trajectory analysis
    print("\n=== Object Type-Specific Trajectory Analysis ===")
    
    # Group trajectories by object type
    trajectories_by_type = defaultdict(list)
    lengths_by_type = defaultdict(list)
    straightness_by_type = defaultdict(list)
    
    for i, obj_type in enumerate(trajectory_object_types):
        trajectories_by_type[obj_type].append(full_trajectories[i])
        lengths_by_type[obj_type].append(traj_lengths[i])
        straightness_by_type[obj_type].append(traj_straightness[i])
    
    # Calculate and store statistics by object type
    object_type_traj_stats = {}
    
    for obj_type, lengths in lengths_by_type.items():
        if len(lengths) > 10:  # Only analyze types with sufficient samples
            type_lengths = np.array(lengths)
            type_straightness = np.array(straightness_by_type[obj_type])
            
            # Calculate statistics
            type_length_mean = np.mean(type_lengths)
            type_length_median = np.median(type_lengths)
            type_straight_mean = np.mean(type_straightness)
            type_straight_median = np.median(type_straightness)
            
            # Calculate percentages
            type_straight_pct = np.sum(type_straightness > 0.9) / len(type_straightness) * 100
            type_curved_pct = np.sum(type_straightness < 0.5) / len(type_straightness) * 100
            
            print(f"{obj_type} trajectory statistics (N={len(lengths)}):")
            print(f"  Mean length: {type_length_mean:.2f}m, Median: {type_length_median:.2f}m")
            print(f"  Mean straightness: {type_straight_mean:.2f}, Median: {type_straight_median:.2f}")
            print(f"  Straight trajectories: {type_straight_pct:.1f}%, Curved: {type_curved_pct:.1f}%")
            
            # Store statistics
            object_type_traj_stats[obj_type] = {
                "count": len(lengths),
                "length_mean": float(type_length_mean),
                "length_median": float(type_length_median),
                "straightness_mean": float(type_straight_mean),
                "straightness_median": float(type_straight_median),
                "straight_percentage": float(type_straight_pct),
                "curved_percentage": float(type_curved_pct)
            }
    
    # Create comparative bar charts
    if len(object_type_traj_stats) > 1:
        # Create bar chart for trajectory lengths
        plt.figure(figsize=(10, 6))
        obj_types = []
        mean_lengths = []
        
        for obj_type, stats in object_type_traj_stats.items():
            obj_types.append(obj_type)
            mean_lengths.append(stats["length_mean"])
        
        y_pos = np.arange(len(obj_types))
        plt.bar(y_pos, mean_lengths, align='center', alpha=0.7)
        plt.xticks(y_pos, obj_types, rotation=45, ha='right')
        plt.ylabel('Mean End-to-end Length (m)')
        plt.title('Average Trajectory Length by Object Type')
        plt.tight_layout()
        plt.savefig(f"{save_dir}/mean_length_by_object_type.png")
        plt.close()
        
        # Create bar chart for straightness
        plt.figure(figsize=(12, 7))
        obj_types = []
        straights = []
        curves = []
        
        for obj_type, stats in object_type_traj_stats.items():
            obj_types.append(obj_type)
            straights.append(stats["straight_percentage"])
            curves.append(stats["curved_percentage"])
        
        x = np.arange(len(obj_types))
        width = 0.35
        
        plt.bar(x - width/2, straights, width, label='Straight (>0.9)', color='blue', alpha=0.7)
        plt.bar(x + width/2, curves, width, label='Curved (<0.5)', color='red', alpha=0.7)
        
        plt.xlabel('Object Type')
        plt.ylabel('Percentage of Trajectories')
        plt.title('Straight vs. Curved Trajectories by Object Type')
        plt.xticks(x, obj_types, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_dir}/straightness_by_object_type.png")
        plt.close()
    
    # Store object type trajectory statistics
    analysis["object_type_analysis"]["trajectory_stats"] = object_type_traj_stats
    
    # 7. FUTURE TRAJECTORY ANALYSIS (if available)
    if has_future:
        print("\n=== Future Trajectory Analysis ===")
        
        # Extract valid future trajectories (matching valid focal agents)
        valid_future_trajectories = []
        future_object_types = []
        
        for i in range(future_data.shape[0]):
            # Check if this focal agent has valid history
            if np.any(focal_valid_mask[i]):
                # Check if future data is valid (non-zero)
                if np.any(np.abs(future_data[i]) > 0):
                    valid_future_trajectories.append(future_data[i])
                    
                    # Store object type
                    if most_common_type[i, 0] >= 0 and most_common_type[i, 0] < len(OBJECT_TYPES):
                        future_object_types.append(OBJECT_TYPES[most_common_type[i, 0]])
                    else:
                        future_object_types.append("unknown")
        
        # Calculate future trajectory properties
        future_lengths = []  # End-to-end length
        future_total_distances = []  # Total distance traveled
        future_straightness = []  # Straightness ratio
        
        for positions in valid_future_trajectories:
            # End-to-end length
            length = np.linalg.norm(positions[-1] - positions[0])
            future_lengths.append(length)
            
            # Total distance
            total_distance = 0
            for j in range(1, len(positions)):
                total_distance += np.linalg.norm(positions[j] - positions[j-1])
            future_total_distances.append(total_distance)
            
            # Straightness ratio
            straightness = length / max(total_distance, 1e-6)
            future_straightness.append(straightness)
        
        # Calculate statistics
        future_length_mean = np.mean(future_lengths)
        future_length_median = np.median(future_lengths)
        future_length_std = np.std(future_lengths)
        future_length_min = np.min(future_lengths)
        future_length_max = np.max(future_lengths)
        
        future_distance_mean = np.mean(future_total_distances)
        future_distance_median = np.median(future_total_distances)
        
        future_straightness_mean = np.mean(future_straightness)
        future_straightness_median = np.median(future_straightness)
        
        # Calculate percentage of straight vs. curved trajectories
        future_straight_pct = np.sum(np.array(future_straightness) > 0.9) / len(future_straightness) * 100
        future_curved_pct = np.sum(np.array(future_straightness) < 0.5) / len(future_straightness) * 100
        
        print(f"Future trajectory statistics (N={len(valid_future_trajectories)}):")
        print(f"  End-to-end length - mean: {future_length_mean:.2f}m, median: {future_length_median:.2f}m, range: [{future_length_min:.2f}, {future_length_max:.2f}]m")
        print(f"  Total distance - mean: {future_distance_mean:.2f}m, median: {future_distance_median:.2f}m")
        print(f"  Straightness ratio - mean: {future_straightness_mean:.2f}, median: {future_straightness_median:.2f}")
        print(f"  Straight trajectories (>0.9): {future_straight_pct:.1f}%")
        print(f"  Curved trajectories (<0.5): {future_curved_pct:.1f}%")
        
        # Store future trajectory statistics
        analysis["trajectory_analysis"]["future"] = {
            "count": len(valid_future_trajectories),
            "end_to_end_distance": {
                "mean": float(future_length_mean),
                "median": float(future_length_median),
                "std": float(future_length_std),
                "min": float(future_length_min),
                "max": float(future_length_max)
            },
            "total_distance": {
                "mean": float(future_distance_mean),
                "median": float(future_distance_median)
            },
            "straightness_ratio": {
                "mean": float(future_straightness_mean),
                "median": float(future_straightness_median),
                "straight_percentage": float(future_straight_pct),
                "curved_percentage": float(future_curved_pct)
            }
        }
        
        # Future vs. History trajectory comparison
        print("\n=== History vs. Future Trajectory Comparison ===")
        
        print(f"History mean length: {length_mean:.2f}m vs. Future mean length: {future_length_mean:.2f}m")
        print(f"History straightness: {straightness_mean:.2f} vs. Future straightness: {future_straightness_mean:.2f}")
        print(f"History straight %: {straight_pct:.1f}% vs. Future straight %: {future_straight_pct:.1f}%")
        print(f"History curved %: {curved_pct:.1f}% vs. Future curved %: {future_curved_pct:.1f}%")
        
        # Future vs. History insight
        if verbose:
            print("\nINSIGHT - Future vs. Historical Trajectories:")
            print(f"- Future trajectories tend to be {'longer' if future_length_mean > length_mean else 'shorter'} than historical ones")
            print(f"  (Future mean: {future_length_mean:.1f}m vs. Historical mean: {length_mean:.1f}m)")
            
            straightness_diff = future_straightness_mean - straightness_mean
            if abs(straightness_diff) < 0.05:
                straightness_change = "similar straightness to"
            elif straightness_diff > 0:
                straightness_change = "straighter than"
            else:
                straightness_change = "more curved than"
                
            print(f"- Future trajectories have {straightness_change} historical ones")
            print(f"  (Future straightness: {future_straightness_mean:.2f} vs. Historical: {straightness_mean:.2f})")
            
            print(f"- Model Implication: The model should account for potential changes in trajectory patterns")
            print(f"  between historical and future timeframes")
        
        # Plot future trajectory length distribution
        plt.figure()
        sns.histplot(future_lengths, bins=50, kde=True)
        plt.axvline(future_length_mean, color='r', linestyle='--', label=f'Mean: {future_length_mean:.2f}')
        plt.axvline(future_length_median, color='g', linestyle='--', label=f'Median: {future_length_median:.2f}')
        plt.xlabel('End-to-end Distance (m)')
        plt.ylabel('Count')
        plt.title('Distribution of Future Trajectory Lengths')
        plt.legend()
        plt.savefig(f"{save_dir}/future_trajectory_lengths.png")
        plt.close()
        
        # Create comparison plot
        plt.figure(figsize=(10, 6))
        sns.kdeplot(traj_lengths, label=f'History (mean={length_mean:.1f}m)', fill=True, alpha=0.4)
        sns.kdeplot(future_lengths, label=f'Future (mean={future_length_mean:.1f}m)', fill=True, alpha=0.4)
        plt.xlabel('End-to-end Distance (m)')
        plt.ylabel('Density')
        plt.title('History vs. Future Trajectory Lengths')
        plt.legend()
        plt.savefig(f"{save_dir}/history_vs_future_lengths.png")
        plt.close()
        
        # Object type-specific future trajectory analysis
        print("\n=== Object Type-Specific Future Trajectory Analysis ===")
        
        # Group trajectories by object type
        future_trajectories_by_type = defaultdict(list)
        future_lengths_by_type = defaultdict(list)
        future_straightness_by_type = defaultdict(list)
        
        for i, obj_type in enumerate(future_object_types):
            future_trajectories_by_type[obj_type].append(valid_future_trajectories[i])
            future_lengths_by_type[obj_type].append(future_lengths[i])
            future_straightness_by_type[obj_type].append(future_straightness[i])
        
        # Calculate and store statistics by object type
        object_type_future_stats = {}
        
        for obj_type, lengths in future_lengths_by_type.items():
            if len(lengths) > 10:  # Only analyze types with sufficient samples
                type_lengths = np.array(lengths)
                type_straightness = np.array(future_straightness_by_type[obj_type])
                
                # Calculate statistics
                type_length_mean = np.mean(type_lengths)
                type_length_median = np.median(type_lengths)
                type_straight_mean = np.mean(type_straightness)
                type_straight_median = np.median(type_straightness)
                
                # Calculate percentages
                type_straight_pct = np.sum(type_straightness > 0.9) / len(type_straightness) * 100
                type_curved_pct = np.sum(type_straightness < 0.5) / len(type_straightness) * 100
                
                print(f"{obj_type} future trajectory statistics (N={len(lengths)}):")
                print(f"  Mean length: {type_length_mean:.2f}m, Median: {type_length_median:.2f}m")
                print(f"  Mean straightness: {type_straight_mean:.2f}, Median: {type_straight_median:.2f}")
                print(f"  Straight trajectories: {type_straight_pct:.1f}%, Curved: {type_curved_pct:.1f}%")
                
                # Store statistics
                object_type_future_stats[obj_type] = {
                    "count": len(lengths),
                    "length_mean": float(type_length_mean),
                    "length_median": float(type_length_median),
                    "straightness_mean": float(type_straight_mean),
                    "straightness_median": float(type_straight_median),
                    "straight_percentage": float(type_straight_pct),
                    "curved_percentage": float(type_curved_pct)
                }
        
        # Store object type future trajectory statistics
        analysis["object_type_analysis"]["future_trajectory_stats"] = object_type_future_stats
        
        # Visualize history and future trajectories for a few samples
        plt.figure(figsize=(15, 12))
        
        # Select random samples
        n_samples = min(12, len(valid_future_trajectories))
        sample_indices = np.random.choice(len(valid_future_trajectories), n_samples, replace=False)
        
        for i, idx in enumerate(sample_indices):
            # Get corresponding scene index
            scene_idx = np.where(focal_valid_mask)[0][idx]
            
            # Get historical trajectory
            scene_valid_mask = focal_valid_mask[scene_idx]
            valid_indices = np.where(scene_valid_mask)[0]
            history_positions = focal_positions[scene_idx, valid_indices]
            
            # Get future trajectory
            future_positions = valid_future_trajectories[idx]
            
            # Plot
            plt.subplot(3, 4, i+1)
            plt.plot(history_positions[:, 0], history_positions[:, 1], 'b-', alpha=0.8, label='History')
            plt.plot(future_positions[:, 0], future_positions[:, 1], 'r-', alpha=0.8, label='Future')
            
            # Mark start and end points
            plt.scatter(history_positions[0, 0], history_positions[0, 1], c='g', marker='o', s=30)
            plt.scatter(history_positions[-1, 0], history_positions[-1, 1], c='b', marker='x', s=40)
            plt.scatter(future_positions[-1, 0], future_positions[-1, 1], c='r', marker='x', s=40)
            
            # Add object type label if available
            obj_type = future_object_types[idx]
            plt.title(f'Sample {i+1}: {obj_type}')
            
            if i == 0:  # Only add legend to first subplot
                plt.legend()
            
            plt.grid(True)
            plt.axis('equal')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/history_future_trajectories.png")
        plt.close()
    
    # 8. SUMMARY AND KEY INSIGHTS
    print("\n=== Summary and Key Insights ===")
    
    # Collect key insights
    insights = {
        "dataset": {
            "total_scenes": int(total_scenes),
            "average_agents_per_scene": float(mean_agents),
            "padding_percentage": float(padding_percentage)
        },
        "position_stats": {
            "recommended_normalization_scale": float(max(x_std, y_std) * 1.5)
        },
        "object_types": {
            "focal_distribution": {k: v for k, v in sorted(focal_object_type_counts.items(), key=lambda x: x[1], reverse=True) if k != "invalid"}
        }
    }
    
    # Add velocity insights if available
    if 'velocity_analysis' in analysis and 'speed_stats' in analysis['velocity_analysis']:
        insights["velocity_stats"] = {
            "recommended_normalization_scale": float(max(vx_std, vy_std) * 3),
            "typical_speed_range": [0, float(speed_percentiles[4])]
        }
    
    # Add object type specific insights
    if 'trajectory_stats' in analysis['object_type_analysis']:
        # Compare mobility patterns between object types
        mobility_patterns = []
        
        # Get three dominant object types
        dominant_types = []
        for obj_type, count in sorted(focal_object_type_counts.items(), key=lambda x: x[1], reverse=True):
            if obj_type != "invalid" and count / total_focal > 0.05:  # More than 5%
                dominant_types.append(obj_type)
            if len(dominant_types) >= 3:
                break
        
        # Add insights for each dominant type
        for obj_type in dominant_types:
            if obj_type in object_type_traj_stats:
                stats = object_type_traj_stats[obj_type]
                mobility_patterns.append({
                    "type": obj_type,
                    "avg_length": stats["length_mean"],
                    "straightness": stats["straightness_mean"],
                    "speed": object_type_speed_stats[obj_type]["speed_mean"] if obj_type in object_type_speed_stats else None,
                    "turning": object_type_angvel_stats[obj_type]["angvel_mean"] if obj_type in object_type_angvel_stats else None
                })
        
        insights["object_type_mobility"] = mobility_patterns
    
    # Add trajectory pattern insights
    if 'trajectory_analysis' in analysis:
        insights["trajectory_patterns"] = {
            "straight_trajectory_percentage": float(straight_pct),
            "curved_trajectory_percentage": float(curved_pct)
        }
    
    # Store insights in analysis dictionary
    analysis["insights"] = insights
    
    # Print key insights
    print(f"Total scenes: {insights['dataset']['total_scenes']}")
    print(f"Average agents per scene: {insights['dataset']['average_agents_per_scene']:.2f}")
    print(f"Padding percentage: {insights['dataset']['padding_percentage']:.1f}%")
    print(f"Recommended position normalization scale: {insights['position_stats']['recommended_normalization_scale']:.2f}m")
    
    if 'velocity_stats' in insights:
        print(f"Recommended velocity normalization scale: {insights['velocity_stats']['recommended_normalization_scale']:.2f} m/s")
        print(f"Typical speed range: 0 to {insights['velocity_stats']['typical_speed_range'][1]:.2f} m/s")
    
    print("\nDominant focal agent types:")
    for obj_type, count in sorted(insights['object_types']['focal_distribution'].items(), key=lambda x: x[1], reverse=True)[:3]:
        print(f"  {obj_type}: {count} agents ({count/total_focal*100:.1f}%)")
    
    if 'trajectory_patterns' in insights:
        print(f"\nTrajectory patterns:")
        print(f"  Straight trajectory percentage: {insights['trajectory_patterns']['straight_trajectory_percentage']:.1f}%")
        print(f"  Curved trajectory percentage: {insights['trajectory_patterns']['curved_trajectory_percentage']:.1f}%")
    
    # Provide comprehensive model design recommendations
    if verbose:
        print("\n=== MODEL DESIGN RECOMMENDATIONS ===")
        print("Based on the exploratory analysis, here are key recommendations for your model design:")
        
        print("\n1. Data Preparation")
        if 'position_stats' in insights and 'recommended_normalization_scale' in insights['position_stats']:
            print(f"   - Use a position normalization scale of ~{insights['position_stats']['recommended_normalization_scale']:.1f}m")
        if 'velocity_stats' in insights and 'recommended_normalization_scale' in insights['velocity_stats']:
            print(f"   - Use a velocity normalization scale of ~{insights['velocity_stats']['recommended_normalization_scale']:.1f} m/s")
        print(f"   - Handle padding properly - {padding_percentage:.1f}% of entries are padding zeros")
        print(f"   - Use valid masks to only process non-padded positions")
        
        print("\n2. Object Type Handling")
        print("   - Include object type as an input feature - different types show distinct movement patterns")
        print("   - Consider separate models or type-specific parameters for dominant types:")
        if 'object_types' in insights and 'focal_distribution' in insights['object_types']:
            for obj_type, count in sorted(insights['object_types']['focal_distribution'].items(), key=lambda x: x[1], reverse=True)[:3]:
                if count > 0 and total_focal > 0:
                    print(f"     * {obj_type}: {count/total_focal*100:.1f}%")
        
        print("\n3. Model Architecture")
        print("   - Use a sequence model (LSTM/GRU) as the base architecture")
        print("   - Incorporate heading information (already available in the data)")
        if ('trajectory_patterns' in insights and 
            'curved_percentage' in insights['trajectory_patterns'] and 
            insights['trajectory_patterns']['curved_percentage'] > 20):
            print("   - Consider a multimodal approach to handle both straight and curved trajectories")
        
        if 'object_type_mobility' in insights:
            print("\n4. Type-Specific Physical Constraints")
            for pattern in insights['object_type_mobility']:
                try:
                    if pattern['turning'] is not None and pattern['speed'] is not None:
                        print(f"   - {pattern['type']}: Speed ~{pattern['speed']:.1f} m/s, " + 
                              f"Turn rate ~{pattern['turning']:.2f} rad/s")
                except (KeyError, TypeError):
                    continue
        
        print("\n5. Training Strategy")
        print("   - Balance the dataset if certain object types are underrepresented")
        print("   - Use a combination of regression loss (e.g., MSE) and physics-based losses")
        print("   - Apply early stopping based on validation metrics")
    
    # Save analysis results to JSON file
    with open(f"{save_dir}/analysis_results.json", 'w') as f:
        # Convert arrays to lists for JSON serialization
        analysis_json = {}
        for key, value in analysis.items():
            if isinstance(value, dict):
                analysis_json[key] = {}
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        analysis_json[key][k] = v.tolist()
                    else:
                        analysis_json[key][k] = v
            elif isinstance(value, np.ndarray):
                analysis_json[key] = value.tolist()
            else:
                analysis_json[key] = value
        
        json.dump(analysis_json, f, indent=2)
    
    print(f"\nExploration complete. Results saved to {save_dir}")
    return analysis

# Example usage
if __name__ == "__main__":
    # Adjust these paths to your actual data paths
    train_path = "/tscc/nfs/home/bax001/scratch/CSE_251B/data/train.npz"
    test_path = "/tscc/nfs/home/bax001/scratch/CSE_251B/data/test_input.npz"
    
    # Explore training data
    print("Exploring training data...")
    train_analysis = explore_trajectory_dataset(train_path, save_dir="train_exploration", verbose=True)
    
    # Explore test data
    print("\nExploring test data...")
    test_analysis = explore_trajectory_dataset(test_path, save_dir="test_exploration", verbose=True)
    
    # Compare train and test distributions
    print("\n=== Train vs Test Comparison ===")
    print(f"Train dataset size: {train_analysis['basic_stats']['total_scenes']}")
    print(f"Test dataset size: {test_analysis['basic_stats']['total_scenes']}")
    
    # Check for distribution shift
    print("\nDistribution shift analysis:")
    
    # Object type distribution shift
    train_focal_types = train_analysis['object_type_analysis']['focal_agents']
    test_focal_types = test_analysis['object_type_analysis']['focal_agents']
    
    print("\nObject type distribution comparison:")
    for obj_type in set(train_focal_types.keys()) | set(test_focal_types.keys()):
        if obj_type != "invalid":
            train_count = train_focal_types.get(obj_type, 0)
            test_count = test_focal_types.get(obj_type, 0)
            
            train_pct = train_count / train_analysis['object_type_analysis']['total_focal_agents'] * 100
            test_pct = test_count / test_analysis['object_type_analysis']['total_focal_agents'] * 100
            
            diff = abs(train_pct - test_pct)
            print(f"  {obj_type}: Train {train_pct:.1f}% vs Test {test_pct:.1f}% (diff: {diff:.1f}%)")
    
    # Final recommendations
    print("\n=== FINAL MODEL RECOMMENDATIONS ===")
    print("Based on the comprehensive data analysis, here are the key recommendations:")
    
    # Data preprocessing
    print("\n1. Data Preprocessing")
    print("   - Handle padding zeros carefully (~50% of the data)")
    print("   - Normalize positions based on the last observation (t=49)")
    print("   - Use separate normalization scales for positions and velocities")
    print("   - Include object type as a one-hot encoded feature")
    
    # Model architecture
    print("\n2. Model Architecture")
    print("   - Implement a multimodal prediction approach with 3-5 modes")
    print("   - Base architecture: LSTM or GRU with 2+ layers for sequence modeling")
    print("   - Include physics-guided components to ensure realistic motion")
    print("   - Create object-type-specific prediction heads or constraints")
    
    # Loss functions
    print("\n3. Loss Functions")
    print("   - Primary regression loss: Minimum ADE/FDE across modes")
    print("   - Physics consistency loss: Penalize unrealistic accelerations/trajectories")
    print("   - Consider weighted losses based on object type importance")
    
    # Training strategy
    print("\n4. Training Strategy")
    print("   - Stratified sampling to balance different object types")
    print("   - Extensive data augmentation (rotations, reflections, speed scaling)")
    print("   - Implement gradual learning rate decay with early stopping")
    print("   - Consider transfer learning techniques between object types")
    
    # Evaluation
    print("\n5. Evaluation Metrics")
    print("   - Primary metrics: minADE and minFDE for multimodal predictions")
    print("   - Secondary metrics: Physical plausibility and mode diversity")
    print("   - Object-type-specific performance analysis")
    
    print("\nImplementing these recommendations should significantly improve your trajectory prediction model.")