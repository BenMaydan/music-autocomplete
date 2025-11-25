
import argparse
import itertools
import json
import os
import sys
import copy
from typing import Dict, List, Any

# Import the trainer module
import trainer

def get_default_config():
    """
    Returns the default configuration dictionary.
    This should match the defaults in trainer.py.
    """
    return {
        'tokenized_dir': "lmd_matched_processed",
        'tokenizer_file': "lmd_matched_tokenizer.json",
        'num_songs': 1000,
        'val_split': 0.1,
        'vocab_size': 5000,
        'block_size': 1024,
        'n_embed': 512,
        'n_blocks': 8,
        'n_head': 8,
        'dropout': 0.1,
        'learning_rate': 3e-4,
        'max_epochs': 100,
        'batch_size': 64,
        'optimizer_type': 'adamw',
        'weight_decay': 0.01,
        'adam_beta1': 0.9,
        'adam_beta2': 0.95,
        'momentum': 0.9,
        'grad_clip': 1.0,
        'scheduler': 'cosine',
        'min_learning_rate': 3e-5,
        'lr_step_size': 10,
        'lr_gamma': 0.1,
        'lr_patience': 3,
        'checkpoint_path': 'trained_models/model.pt',
        'load_checkpoint': False,
        'run_name': None,
        'compile': False
    }

# --- Define Ablation Ranges Here ---

# Phase 1: Architecture Search
# Goal: Find the optimal model size and depth.
ablation_ranges = {
    'n_embed': [128, 256, 512],
    'n_blocks': [2, 4, 8],
    'n_head': [2,4, 8]
}

# Phase 2: Optimization Search (Uncomment when Phase 1 is done)
# Goal: Tune training dynamics for the best architecture.
# ablation_ranges = {
#     'learning_rate': [1e-4, 3e-4, 5e-4],
#     'optimizer_type': ['adamw', 'sgd']
# }

def generate_permutations(base_config: Dict[str, Any], ranges: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """
    Generates all permutations of configuration based on the ranges.
    """
    keys = list(ranges.keys())
    values = list(ranges.values())
    permutations = []

    for combination in itertools.product(*values):
        config = copy.deepcopy(base_config)
        for key, value in zip(keys, combination):
            config[key] = value
        
        # Add a metadata field to track which parameters were changed
        config['_ablation_params'] = {k: v for k, v in zip(keys, combination)}
        permutations.append(config)

    return permutations

def run_ablation_study(study_name: str):
    """
    Runs the ablation study.
    """
    base_config = get_default_config()
    
    if not ablation_ranges:
        print("No ablation ranges defined in 'ablation_ranges'. Please edit the script to add parameters.")
        return

    permutations = generate_permutations(base_config, ablation_ranges)
    num_permutations = len(permutations)

    print(f"--- Ablation Study: {study_name} ---")
    print(f"Parameters to vary: {list(ablation_ranges.keys())}")
    print(f"Total permutations to run: {num_permutations}")
    
    if num_permutations == 0:
        print("No permutations generated. Check your ablation ranges.")
        return

    print("\nPermutations:")
    for i, p in enumerate(permutations):
        print(f"  {i+1}. {p['_ablation_params']}")

    response = input("\nDo you want to continue? (y/n): ")
    if response.lower() != 'y':
        print("Aborting.")
        return

    # Create main study directory
    study_dir = os.path.join("trained_models", study_name)
    os.makedirs(study_dir, exist_ok=True)
    print(f"\nCreated study directory: {study_dir}")

    try:
        for i, config_dict in enumerate(permutations):
            print(f"\n--- Running Permutation {i+1}/{num_permutations} ---")
            print(f"Params: {config_dict['_ablation_params']}")

            # Create a descriptive run name
            # e.g., lr_0.001_bs_32
            run_name = "_".join([f"{k}_{v}" for k, v in config_dict['_ablation_params'].items()])
            
            # Sanitize run name for filesystem
            run_name = run_name.replace(".", "p").replace("/", "_")
            
            # Check if this run is already completed
            full_run_dir = os.path.join(study_dir, run_name)
            history_path = os.path.join(full_run_dir, "loss_history.json")
            
            if os.path.exists(history_path):
                print(f"Run {run_name} already completed. Skipping.")
                continue

            # Update config for this run
            # We need to convert the dict back to a Namespace for the trainer
            # But first, let's set the run_name and checkpoint_path correctly
            
            # The trainer expects checkpoint_path to be the full path to the model file.
            # It then derives the run directory from that.
            # We want: trained_models/<study_name>/<run_name>/model.pt
            
            # FIX: Trainer automatically appends run_name to the directory of checkpoint_path
            # if run_name is provided. So we should NOT include run_name in the base dir here.
            # We want the base dir to be just the study dir.
            checkpoint_path = os.path.join(study_dir, "model.pt")
            
            config_dict['run_name'] = run_name # This is used for logging/display
            config_dict['checkpoint_path'] = checkpoint_path
            
            # Check for existing checkpoint to resume
            # The actual checkpoint file will be at trained_models/<study_name>/<run_name>/model.pt
            # because the trainer constructs the path using run_name.
            actual_checkpoint_path = os.path.join(full_run_dir, "model.pt")
            if os.path.exists(actual_checkpoint_path):
                print(f"Found existing checkpoint at {actual_checkpoint_path}. Resuming training.")
                config_dict['load_checkpoint'] = True
            else:
                config_dict['load_checkpoint'] = False
            
            # Remove our internal metadata before creating Namespace
            ablation_params = config_dict.pop('_ablation_params')
            
            # Create Namespace
            config_namespace = argparse.Namespace(**config_dict)
            
            # Run training
            try:
                train_loss, val_loss = trainer.train_model(config_namespace)
                
                # Save loss history to a JSON file in the run directory
                # The trainer already creates the directory and saves config.json and plots
                # But we ensure it exists just in case (and for tests where trainer is mocked)
                os.makedirs(full_run_dir, exist_ok=True)
                with open(history_path, 'w') as f:
                    json.dump({
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'ablation_params': ablation_params
                    }, f, indent=4)
                print(f"Saved loss history to {history_path}")
                
            except Exception as e:
                print(f"Error running permutation {i+1}: {e}")
                # Decide whether to continue or stop. For now, let's continue.
                continue
                
    except KeyboardInterrupt:
        print("\n\nAblation study interrupted by user.")
        print("You can resume this study later by running the same command.")
        sys.exit(0)
        
    print(f"\n--- Ablation Study {study_name} Completed ---")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ablation_study.py <study_name>")
        sys.exit(1)
    
    study_name = sys.argv[1]
    run_ablation_study(study_name)
