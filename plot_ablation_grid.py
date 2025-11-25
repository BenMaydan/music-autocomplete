
import argparse
import json
import os
import matplotlib.pyplot as plt
import math
from typing import List, Dict, Any

def load_study_data(study_dir: str, x_param: str, y_param: str) -> List[Dict[str, Any]]:
    """
    Loads data from all runs in the study directory.
    Returns a list of dictionaries containing run data.
    """
    runs_data = []
    
    if not os.path.exists(study_dir):
        print(f"Study directory not found: {study_dir}")
        return []

    for run_name in os.listdir(study_dir):
        run_path = os.path.join(study_dir, run_name)
        if not os.path.isdir(run_path):
            continue
            
        history_path = os.path.join(run_path, "loss_history.json")
        if not os.path.exists(history_path):
            continue
            
        try:
            with open(history_path, 'r') as f:
                data = json.load(f)
                
            ablation_params = data.get('ablation_params', {})
            
            # Check if this run has the parameters we are interested in
            if x_param in ablation_params and y_param in ablation_params:
                runs_data.append({
                    'run_name': run_name,
                    'params': ablation_params,
                    'train_loss': data.get('train_loss', []),
                    'val_loss': data.get('val_loss', [])
                })
        except Exception as e:
            print(f"Error loading {run_name}: {e}")
            
    return runs_data

def plot_grid(runs_data: List[Dict[str, Any]], x_param: str, y_param: str, study_dir: str):
    """
    Plots the grid of loss curves.
    """
    if not runs_data:
        print("No matching runs found.")
        return

    # Extract unique values for x and y axes
    x_values = sorted(list(set(d['params'][x_param] for d in runs_data)))
    y_values = sorted(list(set(d['params'][y_param] for d in runs_data)))
    
    n_cols = len(x_values)
    n_rows = len(y_values)
    
    print(f"Grid size: {n_rows} rows x {n_cols} columns")
    print(f"X-axis ({x_param}): {x_values}")
    print(f"Y-axis ({y_param}): {y_values}")

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
    
    # Global title
    fig.suptitle(f"Ablation Study: {x_param} vs {y_param}", fontsize=16)

    for i, y_val in enumerate(y_values):
        for j, x_val in enumerate(x_values):
            ax = axes[i, j]
            
            # Find runs matching this cell
            cell_runs = [r for r in runs_data if r['params'][x_param] == x_val and r['params'][y_param] == y_val]
            
            if not cell_runs:
                ax.text(0.5, 0.5, "No Data", ha='center', va='center')
                continue
                
            for run in cell_runs:
                # Create a label based on OTHER parameters that might be varying
                other_params = {k: v for k, v in run['params'].items() if k not in [x_param, y_param]}
                label_suffix = ", ".join([f"{k}={v}" for k, v in other_params.items()])
                label = f"Run" + (f" ({label_suffix})" if label_suffix else "")
                
                epochs = range(len(run['train_loss']))
                ax.plot(epochs, run['train_loss'], label=f"{label} (Train)", linestyle='--')
                ax.plot(epochs, run['val_loss'], label=f"{label} (Val)")
                
            ax.set_title(f"{x_param}={x_val}, {y_param}={y_val}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.grid(True)
            if len(cell_runs) < 5: # Only show legend if not too crowded
                ax.legend(fontsize='small')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle
    
    output_path = os.path.join(study_dir, f"grid_plot_{x_param}_{y_param}.png")
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Plot ablation study results in a 2D grid.")
    parser.add_argument("study_dir", type=str, help="Path to the ablation study directory.")
    parser.add_argument("x_param", type=str, help="Parameter for the X-axis (columns).")
    parser.add_argument("y_param", type=str, help="Parameter for the Y-axis (rows).")
    
    args = parser.parse_args()
    
    runs_data = load_study_data(args.study_dir, args.x_param, args.y_param)
    plot_grid(runs_data, args.x_param, args.y_param, args.study_dir)

if __name__ == "__main__":
    main()
