import sys
from pathlib import Path
from miditok import REMI, TokenizerConfig
from tqdm import tqdm
import random

# --- 1. CONFIGURATION ---

LMD_FULL_DIR = Path("lmd_matched")
TOKENIZED_DIR = Path("lmd_matched_processed")
TOKENIZER_FILE = Path("lmd_matched_tokenizer.json")

# Create the output directory if it doesn't exist
TOKENIZED_DIR.mkdir(exist_ok=True, parents=True)

# How many files to use for training the tokenizer vocabulary
# 20k is a very robust number for BPE
NUM_FILES_FOR_TOKENIZER_TRAINING = 20000

# --- 2. INITIALIZE & TRAIN THE TOKENIZER ---

if not TOKENIZER_FILE.exists():
    print(f"Tokenizer file not found. Training a new tokenizer...")
    
    # Find all MIDI files
    print("Finding all MIDI files to train tokenizer...")
    # .rglob("*.mid") recursively finds all files ending in .mid
    # This might take a minute
    all_midi_paths = list(LMD_FULL_DIR.rglob("*.mid"))
    print(f"Found {len(all_midi_paths)} total MIDI files.")

    if not all_midi_paths:
        print(f"Error: No MIDI files found in {LMD_FULL_DIR}. Please check the path.")
        sys.exit()

    # Get a random subset for training
    if len(all_midi_paths) > NUM_FILES_FOR_TOKENIZER_TRAINING:
        midi_paths_for_training = random.sample(all_midi_paths, NUM_FILES_FOR_TOKENIZER_TRAINING)
    else:
        midi_paths_for_training = all_midi_paths

    # Initialize the tokenizer (REMI is a great, standard choice)
    # Using default configuration
    tokenizer = REMI()

    print(f"Training tokenizer on {len(midi_paths_for_training)} files... (This can take a while)")
    # Train the tokenizer
    tokenizer.train(
        vocab_size=5000,  # You can adjust this. 5k is a good BPE vocab size.
        files_paths=midi_paths_for_training
    )

    # Save the trained tokenizer
    tokenizer.save(TOKENIZER_FILE)
    print(f"Tokenizer trained and saved to {TOKENIZER_FILE}")

else:
    print(f"Tokenizer file found at {TOKENIZER_FILE}. Loading it.")
    # If the tokenizer file already exists, load it
    tokenizer = REMI(params=TOKENIZER_FILE)
    print("Tokenizer loaded.")

# --- 3. TOKENIZE THE ENTIRE DATASET ---

print("\nTokenizing the entire LMD-full dataset...")
print(f"Input directory: {LMD_FULL_DIR}")
print(f"Output directory: {TOKENIZED_DIR}")

# Use miditok's built-in dataset tokenizer.
# This is highly optimized and uses multi-processing.
# It will find all .mid files, tokenize them, and save them as .json
# in the TOKENIZED_DIR, preserving the original filename.

# Find all MIDI files for tokenization
# We need to pass a sequence (like a list) to the tokenizer, not a generator.
print("\nFinding all MIDI files for final tokenization...")
print(LMD_FULL_DIR)
all_midi_paths_for_tokenization = list(LMD_FULL_DIR.rglob("*.mid"))
print(f"Found {len(all_midi_paths_for_tokenization)} files to tokenize.")

if not all_midi_paths_for_tokenization:
    print(f"Error: No MIDI files found in {LMD_FULL_DIR} to tokenize.")
    sys.exit()

tokenizer.tokenize_dataset(
    files_paths=all_midi_paths_for_tokenization,
    out_dir=TOKENIZED_DIR
)

print("\n--- All done! ---")
print(f"All tokenized JSON files are saved in: {TOKENIZED_DIR}")
print(f"Your tokenizer is saved at: {TOKENIZER_FILE}")
