import json
from pathlib import Path
from tqdm import tqdm
import os

# --- CONFIGURATION ---
# Make sure this matches the directory you're training from.
TOKENIZED_DIR = Path("lmd_matched_processed")
# --- END CONFIGURATION ---

def check_tokenized_files():
    """
    Iterates through all tokenized .json files and checks for:
    1.  0-byte files
    2.  Corrupt JSON
    3.  Empty JSON (e.g., `[]` or `{}`)
    """
    print(f"Scanning directory: {TOKENIZED_DIR.resolve()}")
    
    if not TOKENIZED_DIR.exists():
        print(f"Error: Directory not found: {TOKENIZED_DIR}")
        print("Please make sure you've run the tokenizer.py script and TOKENIZED_DIR is set correctly.")
        return

    # Use .rglob to find all .json files, just in case they are nested
    all_json_files = list(TOKENIZED_DIR.rglob("*.json"))
    
    if not all_json_files:
        print(f"Error: No .json files found in {TOKENIZED_DIR}.")
        print("This is likely the cause of the error. Did the tokenizer.py script run successfully?")
        return
        
    print(f"Found {len(all_json_files)} total .json files. Checking them now...")
    
    bad_files = []
    
    for file_path in tqdm(all_json_files, desc="Checking files"):
        try:
            # 1. Check for 0-byte files
            if file_path.stat().st_size == 0:
                bad_files.append((file_path, "File is 0 bytes"))
                continue
                
            # 2. Check for corrupt JSON or empty data
            with open(file_path, 'r') as f:
                data = json.load(f)
                
                # 3. Check for empty lists `[]` or empty objects `{}`
                if not data:
                    bad_files.append((file_path, "File contains empty data (e.g., [])"))
                    continue

        except json.JSONDecodeError:
            bad_files.append((file_path, "File is corrupt (JSONDecodeError)"))
        except Exception as e:
            bad_files.append((file_path, f"Unknown error: {e}"))

    # --- Print Results ---
    if not bad_files:
        print("\n--- PASSED! ---")
        print("No empty, 0-byte, or corrupt files were found.")
        print("This suggests the problem might be in your `dataloader.py` file,")
        print("in the `__getitem__` method's logic (e.g., slicing a sequence until it's empty).")
    else:
        print(f"\n--- FOUND {len(bad_files)} PROBLEMATIC FILES ---")
        for path, reason in bad_files:
            print(f"  Reason: {reason}")
            print(f"  File:   {path}")
            print("-" * 20)
            
        print("\nRecommendation:")
        print("These files are likely causing your error. The safest thing to do is delete them.")
        print("You can delete them manually or re-run your tokenizer.py to overwrite them.")

if __name__ == "__main__":
    check_tokenized_files()