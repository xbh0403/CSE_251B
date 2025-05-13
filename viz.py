import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
import os
from tqdm import tqdm
import matplotlib.cm as cm

from data_utils.data_utils import TrajectoryDataset
from models.model import Seq2SeqGRUModel

def plot_worst_trajectories(num_worst=200, ego_focus_radius=30.0):
    """
    Generate individual plots for the worst trajectory predictions,
    with separate folders for training and validation sets,
    showing all agents within a relevant distance of the ego vehicle.
    
    Args:
        num_worst: Number of worst trajectories to plot per dataset
        ego_focus_radius: Radius (in meters) around ego trajectory to focus the plot
    """
    # Paths and parameters - updated to handle different environments
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if "tscc" in current_dir:
        train_path = '/tscc/nfs/home/bax001/scratch/CSE_251B/data/train.npz'
        test_path = '/tscc/nfs/home/bax001/scratch/CSE_251B/data/test_input.npz'
    else:
        train_path = 'data/train.npz'
        test_path = 'data/test_input.npz'
    
    scale = 7.0
    batch_size = 16
    hidden_dim = 128
    num_layers = 2
    output_dir = 'worst_trajectories_ego_centered'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    print("Loading the best model...")
    model = Seq2SeqGRUModel(
        input_dim=6,
        hidden_dim=hidden_dim,
        output_seq_len=60,
        output_dim=2,
        num_layers=num_layers
    )
    
    checkpoint = torch.load("best_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Create datasets
    print("Creating datasets...")
    full_train_dataset = TrajectoryDataset(train_path, split='train', scale=scale, augment=False)
    
    # Create train/validation split
    dataset_size = len(full_train_dataset)
    train_size = int(0.9 * dataset_size)
    val_size = dataset_size - train_size
    
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Process each dataset separately
    dataset_names = ['train', 'validation']
    dataset_loaders = [train_loader, val_loader]
    
    # Object type mapping
    type_mapping = ['vehicle', 'pedestrian', 'motorcyclist', 'cyclist', 'bus', 
                   'static', 'background', 'construction', 'riderless_bicycle', 'unknown']
    
    # Process datasets
    for dataset_name, loader in zip(dataset_names, dataset_loaders):
        print(f"\nProcessing {dataset_name} dataset...")
        
        # Create directory for this dataset's worst trajectories
        dataset_dir = os.path.join(output_dir, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Subdirectories for worst by ADE and FDE
        ade_dir = os.path.join(dataset_dir, 'worst_ade')
        fde_dir = os.path.join(dataset_dir, 'worst_fde')
        os.makedirs(ade_dir, exist_ok=True)
        os.makedirs(fde_dir, exist_ok=True)
        
        # List to store trajectories with their errors
        dataset_trajectories = []
        
        # Process the dataset
        scene_idx = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(loader, desc=f"Processing {dataset_name}")):
                # Move data to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(device)
                
                # Skip if no ground truth
                if 'future' not in batch:
                    continue
                
                # Get predictions for focal agent
                predictions = model(batch, teacher_forcing_ratio=0.0)
                
                # Get ground truth
                ground_truth = batch['future']
                
                # Get history data for all agents
                all_history = batch['history']  # [batch, num_agents, seq_len, feature_dim]
                
                # Calculate error for each sample in the batch (in normalized space)
                for i in range(ground_truth.shape[0]):
                    # Calculate Average Displacement Error (ADE)
                    error = torch.sqrt(torch.sum((predictions[i] - ground_truth[i]) ** 2, dim=1))
                    ade = torch.mean(error).item()
                    
                    # Calculate Final Displacement Error (FDE)
                    fde = torch.sqrt(torch.sum((predictions[i, -1] - ground_truth[i, -1]) ** 2)).item()
                    
                    # Extract and denormalize all agents' history for this scene
                    scene_all_agents = all_history[i]  # [num_agents, seq_len, feature_dim]
                    
                    # Create a valid mask to filter out padding agents (all zeros)
                    valid_mask = torch.sum(torch.abs(scene_all_agents[:, :, :2]), dim=(1, 2)) > 0
                    valid_agents = scene_all_agents[valid_mask]
                    
                    # Get the scale and origin for denormalization
                    scale_factor = batch['scale'][i].item()
                    origin = batch['origin'][i].cpu().numpy()
                    
                    # Denormalize all agents' positions
                    valid_agents_pos = valid_agents[:, :, :2] * scale_factor
                    
                    # Extract focal agent history (already denormalized above)
                    focal_history = valid_agents_pos[0].cpu().numpy()
                    
                    # Denormalize predictions and ground truth
                    pred_denorm = (predictions[i].cpu().numpy() * scale_factor) + origin
                    gt_denorm = (ground_truth[i].cpu().numpy() * scale_factor) + origin
                    
                    # CRITICAL FIX: Ensure the first prediction and ground truth points
                    # exactly match the last history point in absolute coordinates
                    # For the history, add back the origin to get absolute coordinates
                    focal_history_absolute = focal_history + origin
                    
                    # Get types for valid agents
                    agent_types = []
                    if scene_all_agents.shape[-1] > 5:  # If object type is included
                        for agent_idx in range(valid_agents.shape[0]):
                            agent_type_idx = int(valid_agents[agent_idx, -1, 5].item())
                            if 0 <= agent_type_idx < len(type_mapping):
                                agent_types.append(type_mapping[agent_type_idx])
                            else:
                                agent_types.append("unknown")
                    else:
                        agent_types = ["unknown"] * valid_agents.shape[0]
                    
                    # Store focal agent object type
                    focal_object_type = agent_types[0] if agent_types else "unknown"
                    
                    # Other agents' histories (already denormalized above)
                    other_agents_history = [pos.cpu().numpy() for pos in valid_agents_pos[1:]]
                    
                    # Add origin to get absolute coordinates for other agents
                    other_agents_absolute = [hist + origin for hist in other_agents_history]
                    
                    other_agents_types = agent_types[1:] if len(agent_types) > 1 else []
                    
                    # Store the trajectory data and error
                    dataset_trajectories.append({
                        'scene_idx': scene_idx,
                        'history': focal_history_absolute,  # Now in absolute coordinates
                        'prediction': pred_denorm,          # Now in absolute coordinates
                        'ground_truth': gt_denorm,          # Now in absolute coordinates
                        'origin': origin,
                        'other_agents': other_agents_absolute,  # Now in absolute coordinates
                        'other_agents_types': other_agents_types,
                        'ade': ade,
                        'fde': fde,
                        'object_type': focal_object_type
                    })
                    
                    scene_idx += 1
        
        print(f"Processed {len(dataset_trajectories)} trajectories from {dataset_name} dataset")
        
        # Sort trajectories by ADE (worst first)
        sorted_by_ade = sorted(dataset_trajectories, key=lambda x: x['ade'], reverse=True)
        
        # Select the worst N trajectories by ADE
        worst_by_ade = sorted_by_ade[:num_worst]
        
        # Plot the worst trajectories by ADE
        print(f"Plotting {num_worst} worst trajectories by ADE for {dataset_name}...")
        
        # Summary file for worst ADE trajectories
        with open(os.path.join(ade_dir, 'summary.txt'), 'w') as f:
            f.write(f"Summary of {num_worst} Worst Trajectories by ADE in {dataset_name} dataset\n")
            f.write("=" * 50 + "\n\n")
            
            for rank, traj in enumerate(tqdm(worst_by_ade, desc=f"Creating ADE plots for {dataset_name}")):
                # Create plot
                fig, ax = plt.subplots(figsize=(14, 12))
                
                # Extract trajectory components - all are now in absolute coordinates
                history = traj['history']
                prediction = traj['prediction']
                ground_truth = traj['ground_truth']
                other_agents = traj['other_agents']
                other_agents_types = traj['other_agents_types']
                
                # VALIDATION CHECK: Print the last history point and first future points
                # to verify they match
                if rank == 0:
                    print(f"VALIDATION - Last history point: {history[-1]}")
                    print(f"VALIDATION - First prediction point: {prediction[0]}")
                    print(f"VALIDATION - First ground truth point: {ground_truth[0]}")
                
                # Focus on ego vehicle trajectory
                # First, determine the bounding box of the ego vehicle's trajectory
                ego_points = np.vstack([history, prediction, ground_truth])
                ego_min_x, ego_min_y = np.min(ego_points, axis=0)
                ego_max_x, ego_max_y = np.max(ego_points, axis=0)
                
                # Calculate center and range of ego trajectory
                ego_center_x = (ego_min_x + ego_max_x) / 2
                ego_center_y = (ego_min_y + ego_max_y) / 2
                ego_range_x = max(ego_max_x - ego_min_x, 10.0)  # Ensure minimum width
                ego_range_y = max(ego_max_y - ego_min_y, 10.0)  # Ensure minimum height
                
                # Set boundaries focused on ego vehicle with buffer
                min_x = ego_center_x - ego_focus_radius
                max_x = ego_center_x + ego_focus_radius
                min_y = ego_center_y - ego_focus_radius
                max_y = ego_center_y + ego_focus_radius
                
                # Filter other agents to only include those within or near the ego vehicle's area
                nearby_agents = []
                nearby_agent_types = []
                distant_agent_count = 0
                
                for agent_idx, (agent_hist, agent_type) in enumerate(zip(other_agents, other_agents_types)):
                    # Calculate if any part of this agent's trajectory is within our view bounds
                    # Add a small buffer (5m) to include agents that are just outside but still relevant
                    buffer = 5.0
                    agent_in_view = np.any(
                        (agent_hist[:, 0] >= min_x - buffer) & 
                        (agent_hist[:, 0] <= max_x + buffer) & 
                        (agent_hist[:, 1] >= min_y - buffer) & 
                        (agent_hist[:, 1] <= max_y + buffer)
                    )
                    
                    if agent_in_view:
                        nearby_agents.append(agent_hist)
                        nearby_agent_types.append(agent_type)
                    else:
                        distant_agent_count += 1
                
                # Plot nearby agents first (in the background)
                if nearby_agents:
                    colors = cm.tab10(np.linspace(0, 1, min(10, len(nearby_agents))))
                    for agent_idx, (agent_hist, agent_type) in enumerate(zip(nearby_agents, nearby_agent_types)):
                        color_idx = agent_idx % len(colors)
                        ax.plot(agent_hist[:, 0], agent_hist[:, 1], '-', color=colors[color_idx], 
                                alpha=0.5, linewidth=1.5, 
                                label=f'Agent {agent_idx+2}: {agent_type}' if agent_idx < 5 else "")
                        
                        # Mark start and end points
                        ax.scatter(agent_hist[0, 0], agent_hist[0, 1], color=colors[color_idx], s=30, marker='o', alpha=0.5)
                        ax.scatter(agent_hist[-1, 0], agent_hist[-1, 1], color=colors[color_idx], s=30, marker='x', alpha=0.5)
                
                # Plot focal agent history
                ax.plot(history[:, 0], history[:, 1], 'b-', linewidth=2.5, label=f'Focal Agent: {traj["object_type"]}')
                ax.scatter(history[0, 0], history[0, 1], c='b', s=100, marker='o', label='Start')
                ax.scatter(history[-1, 0], history[-1, 1], c='b', s=100, marker='x', label='History End')
                
                # Plot ground truth future
                ax.plot(ground_truth[:, 0], ground_truth[:, 1], 'g-', linewidth=2.5, label='Ground Truth Future')
                ax.scatter(ground_truth[-1, 0], ground_truth[-1, 1], c='g', s=100, marker='x', label='GT End')
                
                # Plot predicted future
                ax.plot(prediction[:, 0], prediction[:, 1], 'r-', linewidth=2.5, label='Predicted Future')
                ax.scatter(prediction[-1, 0], prediction[-1, 1], c='r', s=100, marker='x', label='Pred End')
                
                # Set axis limits to focus on ego vehicle
                ax.set_xlim(min_x, max_x)
                ax.set_ylim(min_y, max_y)
                
                # Add grid and labels
                ax.grid(True)
                ax.set_aspect('equal')
                ax.set_xlabel('X Position (m)', fontsize=12)
                ax.set_ylabel('Y Position (m)', fontsize=12)
                
                # Add error metrics to title, including note about distant agents if any were excluded
                title = f"{dataset_name.capitalize()} - Rank {rank+1}: ADE={traj['ade']:.2f}m, FDE={traj['fde']:.2f}m\n"
                title += f"Focal: {traj['object_type']}, Nearby Agents: {len(nearby_agents)}"
                if distant_agent_count > 0:
                    title += f" (+{distant_agent_count} distant agents not shown)"
                ax.set_title(title, fontsize=14)
                
                # Add legend (limit to focal agent + first few surrounding agents)
                handles, labels = ax.get_legend_handles_labels()
                # Keep focal agent entries and limit surrounding agents
                if len(handles) > 7:  # Focal agent has 4 entries, limit to 3 surrounding agents
                    keep_indices = list(range(4)) + list(range(4, 7))  # Keep first 4 + 3 more
                    handles = [handles[i] for i in keep_indices]
                    labels = [labels[i] for i in keep_indices]
                    # Add a note about omitted agents
                    handles.append(plt.Line2D([0], [0], color='gray', linestyle='-', alpha=0.5))
                    labels.append(f"+ {len(nearby_agents) - 3} more nearby agents")
                
                ax.legend(handles, labels, fontsize=10, loc='best')
                
                # Save plot
                fig.tight_layout()
                fig.savefig(os.path.join(ade_dir, f'rank_{rank+1:03d}_ade_{traj["ade"]:.2f}_fde_{traj["fde"]:.2f}.png'), dpi=300)
                plt.close(fig)
                
                # Write to summary file
                f.write(f"Rank {rank+1}:\n")
                f.write(f"  Scene Index: {traj['scene_idx']}\n")
                f.write(f"  Focal Agent Type: {traj['object_type']}\n")
                f.write(f"  Number of Nearby Agents: {len(nearby_agents)}\n")
                if distant_agent_count > 0:
                    f.write(f"  Number of Distant Agents (not shown): {distant_agent_count}\n")
                if nearby_agent_types:
                    type_counts = {}
                    for t in nearby_agent_types:
                        if t not in type_counts:
                            type_counts[t] = 0
                        type_counts[t] += 1
                    f.write(f"  Nearby Agent Types: {', '.join([f'{k}({v})' for k, v in type_counts.items()])}\n")
                f.write(f"  ADE: {traj['ade']:.4f} m\n")
                f.write(f"  FDE: {traj['fde']:.4f} m\n")
                f.write("\n")
        
        # The rest of the code for FDE visualizations would need the same fixes...

if __name__ == "__main__":
    # Use a 30-meter radius around the ego vehicle - adjust this value as needed
    plot_worst_trajectories(num_worst=200, ego_focus_radius=30.0)