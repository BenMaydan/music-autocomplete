import sys
import subprocess
from pathlib import Path
import shutil
from tqdm import tqdm
import time

# --- 1. CONFIGURATION ---
LMD_FULL_DIR = Path("lmd_full") # Your original dataset
LMD_CLEAN_DIR = Path("lmd_full_CLEAN") 
LOG_FILE = Path("corrupt_files.log")
WORKER_SCRIPT = Path("datacleaner_worker.py")
# -----------------------------

if not LMD_FULL_DIR.exists():
    print(f"Error: Source directory not found at {LMD_FULL_DIR}")
    sys.exit()

if not WORKER_SCRIPT.exists():
    print(f"Error: Worker script not found at {WORKER_SCRIPT}")
    print("Please create 'datacleaner_worker.py' in the same directory.")
    sys.exit()

LMD_CLEAN_DIR.mkdir(exist_ok=True)

print("Finding all MIDI files...")
all_files = list(LMD_FULL_DIR.rglob("*.mid"))
total_files = len(all_files)
print(f"Found {total_files} files. Starting automated validation...")

good_count = 0
corrupt_count = 0
start_time = time.time()

# We use 'a' (append) mode to be resumable
with open(LOG_FILE, 'a', encoding='utf-8') as log:
    if log.tell() == 0:
        log.write(f"--- Corrupt MIDI Files Log for {LMD_FULL_DIR} ---\n")

    for file_path in tqdm(all_files, desc="Validating MIDI files"):
        relative_path = file_path.relative_to(LMD_FULL_DIR)
        output_path = LMD_CLEAN_DIR / relative_path

        # This makes the script resumable!
        if output_path.exists():
            good_count += 1
            continue

        try:
            # --- THIS IS THE NEW PART ---
            # Run the worker script as a separate process
            # We pass the file path as an argument
            result = subprocess.run(
                [sys.executable, str(WORKER_SCRIPT), str(file_path)],
                capture_output=True, text=True, timeout=2
            )
            
            # --- Check the result ---
            if result.returncode == 0:
                # Success! Copy the file
                output_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file_path, output_path)
                good_count += 1
            else:
                # Failure. The worker exited with an error or crashed
                corrupt_count += 1
                error_message = result.stderr.strip()
                if not error_message:
                    error_message = f"Hard crash (malloc error) or timeout."
                
                log.write(f"{relative_path} - Error: {error_message}\n")

        except subprocess.TimeoutExpired:
            # The file is so bad it caused the worker to hang
            corrupt_count += 1
            log.write(f"{relative_path} - Error: Processing timed out (hung process)\n")
        except Exception as e:
            # Should not happen, but good to catch
            corrupt_count += 1
            log.write(f"{relative_path} - Error: Main script error: {e}\n")


# --- 6. FINAL REPORT ---
end_time = time.time()
elapsed_time = end_time - start_time
print("\n--- Cleaning Complete ---")
print(f"Total time taken: {elapsed_time / 60:.2f} minutes")
print(f"Total files processed: {total_files}")
print(f"Good files copied: {good_count}")
print(f"Corrupt/Bad files skipped: {corrupt_count}")
print(f"\nLog of skipped files saved to: {LOG_FILE}")
print(f"Your clean dataset is at: {LMD_CLEAN_DIR.resolve()}")