import torch
import argparse
import os
import shutil

def fix_checkpoint(checkpoint_path, backup=True):
    """
    Loads a checkpoint, removes '_orig_mod.' prefixes from keys caused by 
    torch.compile(), and saves it back.
    """
    print(f"Processing checkpoint: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: File {checkpoint_path} not found.")
        return

    # 1. Load the checkpoint
    # map_location='cpu' ensures we can load it even if we don't have the same GPU setup active
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return

    # 2. Locate the state_dict
    # Sometimes checkpoints are just the state_dict, sometimes they are a dict containing 'model_state_dict'
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        is_nested = True
    else:
        state_dict = checkpoint
        is_nested = False

    # 3. Create a new state dict with fixed keys
    new_state_dict = {}
    fixed_count = 0
    
    for key, value in state_dict.items():
        if key.startswith('_orig_mod.'):
            new_key = key.replace('_orig_mod.', '')
            new_state_dict[new_key] = value
            fixed_count += 1
        else:
            new_state_dict[key] = value

    if fixed_count == 0:
        print("No keys with '_orig_mod.' prefix found. The checkpoint might already be clean.")
        return

    print(f"Fixed {fixed_count} keys.")

    # 4. Backup the original file
    if backup:
        backup_path = checkpoint_path + ".bak"
        shutil.copy2(checkpoint_path, backup_path)
        print(f"Original checkpoint backed up to: {backup_path}")

    # 5. Save the modified checkpoint
    if is_nested:
        checkpoint['model_state_dict'] = new_state_dict
    else:
        checkpoint = new_state_dict

    torch.save(checkpoint, checkpoint_path)
    print(f"Successfully saved fixed checkpoint to: {checkpoint_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix torch.compile prefixes in checkpoints")
    parser.add_argument("path", type=str, help="Path to the .pt model file")
    args = parser.parse_args()

    fix_checkpoint(args.path)