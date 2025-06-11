#!/usr/bin/env python
"""
Training metrics analysis tool

This script provides functionality to analyze training metrics from saved log files.
It can compare multiple model runs, visualize training curves, and generate insightful
comparisons to help improve model performance.

Usage:
    python analyze_training.py [--logs_dir LOGS_DIR] [--models MODEL_NAMES [MODEL_NAMES ...]]

Example:
    python analyze_training.py --logs_dir logs --models gru_model multimodal_gru_model
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import json
from datetime import datetime

def load_metrics_from_json(json_path):
    """Load metrics history from a JSON file"""
    with open(json_path, 'r') as f:
        metrics = json.load(f)
    return metrics

def load_metrics_from_csv(csv_path):
    """Load metrics history from a CSV file"""
    return pd.read_csv(csv_path)

def find_model_logs(logs_dir, model_name=None):
    """Find all log files for a specific model or all models if None"""
    if model_name:
        json_pattern = os.path.join(logs_dir, f"{model_name}_*_metrics.json")
        csv_pattern = os.path.join(logs_dir, f"{model_name}_*_metrics.csv")
    else:
        json_pattern = os.path.join(logs_dir, "*_metrics.json")
        csv_pattern = os.path.join(logs_dir, "*_metrics.csv")
        
    json_files = glob.glob(json_pattern)
    csv_files = glob.glob(csv_pattern)
    
    return {
        'json': json_files,
        'csv': csv_files
    }

def plot_training_curves(metrics_df, model_name, output_dir=None):
    """Plot various training curves from the metrics dataframe"""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Loss curves
    plt.figure(figsize=(12, 8))
    plt.plot(metrics_df['epoch'], metrics_df['train_loss'], label='Training Loss')
    plt.plot(metrics_df['epoch'], metrics_df['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} - Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, f'{model_name}_loss_curves.png'), dpi=300)
    else:
        plt.show()
    plt.close()
    
    # Error metrics
    plt.figure(figsize=(12, 8))
    if 'train_mse_unnorm' in metrics_df.columns:
        plt.plot(metrics_df['epoch'], metrics_df['train_mse_unnorm'], label='Train MSE (unnorm)')
    plt.plot(metrics_df['epoch'], metrics_df['val_mse_unnorm'], label='Val MSE (unnorm)')
    plt.plot(metrics_df['epoch'], metrics_df['val_mae'], label='Val MAE')
    plt.xlabel('Epoch')
    plt.ylabel('Error (meters)')
    plt.title(f'{model_name} - Error Metrics')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, f'{model_name}_error_metrics.png'), dpi=300)
    else:
        plt.show()
    plt.close()
    
    # Learning rate
    if 'learning_rate' in metrics_df.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(metrics_df['epoch'], metrics_df['learning_rate'])
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title(f'{model_name} - Learning Rate Schedule')
        plt.yscale('log')
        plt.grid(True)
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, f'{model_name}_learning_rate.png'), dpi=300)
        else:
            plt.show()
        plt.close()
    
    # Physics loss for multimodal models
    if 'train_physics_loss' in metrics_df.columns and 'val_physics_loss' in metrics_df.columns:
        plt.figure(figsize=(12, 8))
        plt.plot(metrics_df['epoch'], metrics_df['train_physics_loss'], label='Train Physics Loss')
        plt.plot(metrics_df['epoch'], metrics_df['val_physics_loss'], label='Val Physics Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Physics Loss')
        plt.title(f'{model_name} - Physics Constraint Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, f'{model_name}_physics_loss.png'), dpi=300)
        else:
            plt.show()
        plt.close()

def compare_models(log_files, output_dir=None):
    """Compare multiple model runs with various metrics"""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load all model metrics
    model_metrics = {}
    for log_file in log_files:
        if log_file.endswith('.json'):
            metrics = load_metrics_from_json(log_file)
            model_name = os.path.basename(log_file).split('_metrics.json')[0]
            model_metrics[model_name] = pd.DataFrame(metrics)
        elif log_file.endswith('.csv'):
            metrics = load_metrics_from_csv(log_file)
            model_name = os.path.basename(log_file).split('_metrics.csv')[0]
            model_metrics[model_name] = metrics
    
    if not model_metrics:
        print("No metrics found to compare.")
        return
    
    # Find best epochs and validation metrics for each model
    best_metrics = {}
    for model_name, metrics_df in model_metrics.items():
        best_epoch_idx = metrics_df['val_loss'].idxmin()
        best_metrics[model_name] = {
            'best_epoch': metrics_df.loc[best_epoch_idx, 'epoch'],
            'val_loss': metrics_df.loc[best_epoch_idx, 'val_loss'],
            'val_mae': metrics_df.loc[best_epoch_idx, 'val_mae'],
            'val_mse': metrics_df.loc[best_epoch_idx, 'val_mse_unnorm'] 
                if 'val_mse_unnorm' in metrics_df.columns 
                else metrics_df.loc[best_epoch_idx, 'val_mse'],
            'training_time': len(metrics_df)  # Approximation by number of epochs
        }
    
    # Create comparison table
    comparison_df = pd.DataFrame(best_metrics).T
    print("\nModel Comparison - Best Validation Metrics:")
    print(comparison_df.to_string())
    
    if output_dir:
        comparison_path = os.path.join(output_dir, "model_comparison.csv")
        comparison_df.to_csv(comparison_path)
        print(f"Comparison saved to: {comparison_path}")
    
    # Plot comparisons
    # 1. Validation Loss Comparison
    plt.figure(figsize=(14, 8))
    for model_name, metrics_df in model_metrics.items():
        plt.plot(metrics_df['epoch'], metrics_df['val_loss'], label=f'{model_name}')
    
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss Comparison')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, "val_loss_comparison.png"), dpi=300)
    else:
        plt.show()
    plt.close()
    
    # 2. Validation MAE Comparison
    plt.figure(figsize=(14, 8))
    for model_name, metrics_df in model_metrics.items():
        plt.plot(metrics_df['epoch'], metrics_df['val_mae'], label=f'{model_name}')
    
    plt.xlabel('Epoch')
    plt.ylabel('Validation MAE (meters)')
    plt.title('Validation MAE Comparison')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, "val_mae_comparison.png"), dpi=300)
    else:
        plt.show()
    plt.close()
    
    # 3. Training Progress Comparison (relative to best val loss)
    plt.figure(figsize=(14, 8))
    for model_name, metrics_df in model_metrics.items():
        best_val = metrics_df['val_loss'].min()
        relative_vals = metrics_df['val_loss'] / best_val
        plt.plot(metrics_df['epoch'], relative_vals, label=f'{model_name}')
    
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Epoch')
    plt.ylabel('Relative Validation Loss (1.0 = best)')
    plt.title('Training Convergence Comparison')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, "convergence_comparison.png"), dpi=300)
    else:
        plt.show()
    plt.close()
    
    return comparison_df

def analyze_training_history(metrics_df, model_name, output_dir=None):
    """Analyze training dynamics and generate insights"""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    insights = []
    
    # Check for overfitting
    best_epoch = metrics_df['val_loss'].idxmin()
    best_epoch_num = metrics_df.loc[best_epoch, 'epoch']
    final_epoch = metrics_df['epoch'].max()
    
    # Training vs validation loss gap
    train_val_gap = metrics_df['train_loss'].min() / metrics_df['val_loss'].min()
    
    insights.append(f"Model reached best validation at epoch {best_epoch_num}/{final_epoch}")
    
    if train_val_gap < 0.5:
        insights.append("ISSUE: Large gap between training and validation loss suggests overfitting")
        insights.append("SUGGESTION: Increase regularization (weight decay) or dropout")
    
    # Learning rate analysis
    if 'learning_rate' in metrics_df.columns:
        lr_at_best = metrics_df.loc[best_epoch, 'learning_rate']
        if lr_at_best == metrics_df['learning_rate'].min():
            insights.append(f"Best performance reached at minimum learning rate ({lr_at_best:.6f})")
            insights.append("SUGGESTION: Try longer training with slower decay or smaller minimum LR")
        elif lr_at_best == metrics_df['learning_rate'].max():
            insights.append(f"Best performance reached at initial learning rate ({lr_at_best:.6f})")
            insights.append("SUGGESTION: Training may need more epochs to converge")
    
    # Convergence speed
    epochs_to_half = None
    best_val_loss = metrics_df['val_loss'].min()
    initial_val_loss = metrics_df['val_loss'].iloc[0]
    target_val_loss = (initial_val_loss + best_val_loss) / 2
    
    for i, val_loss in enumerate(metrics_df['val_loss']):
        if val_loss <= target_val_loss:
            epochs_to_half = i
            break
    
    if epochs_to_half is not None:
        epochs_to_half_actual = metrics_df.iloc[epochs_to_half]['epoch']
        insights.append(f"Model reached 50% of improvement in {epochs_to_half_actual} epochs")
        
        if epochs_to_half_actual < 5:
            insights.append("INSIGHT: Very fast initial convergence may indicate room for higher learning rate")
        elif epochs_to_half_actual > 20:
            insights.append("INSIGHT: Slow convergence may benefit from higher initial learning rate")
    
    # Error metrics analysis
    if all(x in metrics_df.columns for x in ['val_mae', 'val_mse_unnorm']):
        final_mae = metrics_df.loc[best_epoch, 'val_mae']
        final_mse = metrics_df.loc[best_epoch, 'val_mse_unnorm']
        final_rmse = np.sqrt(final_mse)
        
        insights.append(f"Best model metrics - MAE: {final_mae:.4f}m, RMSE: {final_rmse:.4f}m")
        
        # Check if MAE << RMSE, which would indicate outliers
        if final_mae < 0.5 * final_rmse:
            insights.append("INSIGHT: Large difference between MAE and RMSE suggests outlier predictions")
            insights.append("SUGGESTION: Consider robust loss functions or addressing edge cases")
    
    # Physics loss analysis for multimodal models
    if all(x in metrics_df.columns for x in ['train_physics_loss', 'val_physics_loss']):
        phys_at_best = metrics_df.loc[best_epoch, 'val_physics_loss']
        initial_phys = metrics_df['val_physics_loss'].iloc[0]
        
        if phys_at_best > 0.9 * initial_phys:
            insights.append("INSIGHT: Physics loss barely improved, may need stronger physics weight")
        elif phys_at_best < 0.2 * initial_phys:
            insights.append("INSIGHT: Significant physics loss improvement")
    
    # Print and save insights
    print("\n=== TRAINING ANALYSIS INSIGHTS ===")
    for insight in insights:
        print(f"- {insight}")
    
    if output_dir:
        with open(os.path.join(output_dir, f"{model_name}_insights.txt"), 'w') as f:
            f.write("=== TRAINING ANALYSIS INSIGHTS ===\n")
            for insight in insights:
                f.write(f"- {insight}\n")
    
    return insights

def main():
    parser = argparse.ArgumentParser(description='Analyze training metrics from logs')
    parser.add_argument('--logs_dir', type=str, default='logs', help='Directory containing log files')
    parser.add_argument('--models', nargs='+', help='Specific model names to analyze')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save analysis results')
    parser.add_argument('--compare_only', action='store_true', help='Only compare models, skip individual analysis')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.logs_dir):
        print(f"Error: Logs directory '{args.logs_dir}' not found.")
        return
    
    # Create output directory with timestamp if specified
    output_dir = None
    if args.output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(args.output_dir, f"analysis_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        print(f"Results will be saved to: {output_dir}")
    
    # Find log files
    if args.models:
        # Analyze specific models
        all_log_files = []
        for model_name in args.models:
            model_logs = find_model_logs(args.logs_dir, model_name)
            all_log_files.extend(model_logs['csv'])
            
            if not args.compare_only:
                print(f"\nAnalyzing model: {model_name}")
                for csv_file in model_logs['csv']:
                    metrics_df = pd.read_csv(csv_file)
                    model_output_dir = os.path.join(output_dir, model_name) if output_dir else None
                    
                    # Individual model analysis
                    plot_training_curves(metrics_df, model_name, model_output_dir)
                    analyze_training_history(metrics_df, model_name, model_output_dir)
    else:
        # Find all models in the logs directory
        all_logs = find_model_logs(args.logs_dir)
        all_log_files = all_logs['csv']
        
        if not args.compare_only:
            print(f"\nAnalyzing all models in {args.logs_dir}")
            for csv_file in all_log_files:
                model_name = os.path.basename(csv_file).split('_metrics.csv')[0]
                metrics_df = pd.read_csv(csv_file)
                model_output_dir = os.path.join(output_dir, model_name) if output_dir else None
                
                # Individual model analysis
                plot_training_curves(metrics_df, model_name, model_output_dir)
                analyze_training_history(metrics_df, model_name, model_output_dir)
    
    # Compare all models found
    if len(all_log_files) > 1:
        print("\nComparing models:")
        for csv_file in all_log_files:
            print(f"- {os.path.basename(csv_file).split('_metrics.csv')[0]}")
        
        compare_models(all_log_files, output_dir)
    else:
        print("\nNot enough models found for comparison (need at least 2).")

if __name__ == "__main__":
    main()