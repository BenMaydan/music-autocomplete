import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from miditok import REMI
from pathlib import Path
import random
import math
import sys
import json

# --- 1. NEW CUSTOM DATASET ---
# This replaces `miditok.DatasetMIDI`

class MIDITokenDataset(Dataset):
    """
    A simple, robust Dataset for loading tokenized JSON files.
    It does not silently filter files.
    """
    def __init__(self, files_paths: list[Path], max_seq_len: int):
        self.files_paths = files_paths
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.files_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        file_path = self.files_paths[idx]
        
        try:
            with open(file_path, 'r') as f:
                data_dict = json.load(f) # Renamed to 'data_dict' for clarity
        except Exception as e:
            # This will be caught by the DataLoader worker
            raise IOError(f"Error loading file {file_path}: {e}")

        # --- START OF FIX (v2) ---
        # We need to extract the token list from the correct key.
        
        if isinstance(data_dict, dict):
            # The default key miditok uses is "ids"
            if "ids" in data_dict:
                tokens_2d = data_dict["ids"] # This is a 2D list (list of tracks)
            elif "tokens" in data_dict: # Check for "tokens" as a fallback
                tokens_2d = data_dict["tokens"]
            else:
                raise KeyError(f"Loaded JSON from {file_path} as dict, but key 'ids' or 'tokens' not found. Found keys: {data_dict.keys()}")
        elif isinstance(data_dict, list):
             # This handles the case where the JSON is just a flat list (or 2D list)
             tokens_2d = data_dict
        else:
            raise TypeError(f"Loaded JSON from {file_path} is not a dict or list, but {type(data_dict)}.")
        
        # --- NEW FLATTENING STEP ---
        # `tokens_2d` is a list of tracks (lists). We flatten it into one sequence.
        # We must also handle that `tokens_2d` could be `[]` (empty file)
        # or `[[]]` (empty track).
        if not tokens_2d:
             print(f"\n[Data Warning] Skipping file (no tracks): {file_path}\n")
             raise IndexError(f"File {file_path} has no tracks.")
        
        # This list comprehension flattens the 2D list into a 1D list
        tokens_1d = [token for track in tokens_2d for token in track]
        # --- END OF FIX (v2) ---


        # CRITICAL CHECK: Ensure the *flattened* sequence has at least 2 tokens
        if not tokens_1d or len(tokens_1d) < 2:
            # Print a warning, and raise an error that the DataLoader will catch
            # This will cause the worker to try and fetch a *different* file
            print(f"\n[Data Warning] Skipping file (empty or too short after flattening): {file_path}\n")
            raise IndexError(f"File {file_path} is empty or too short after flattening.")
        
        # Pass the 1D list to torch.tensor
        data = torch.tensor(tokens_1d, dtype=torch.long)
        
        # Truncate to max_seq_len + 1 (for x and y)
        data = data[:self.max_seq_len + 1]
        
        # Create input (x) and target (y)
        x = data[:-1]
        y = data[1:]
        
        return x, y

# --- 2. NEW CUSTOM COLLATOR ---
# This replaces `miditok.DataCollator`

class CustomDataCollator:
    """
    A robust collate function that handles padding.
    """
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
        # `batch` is a list of (x, y) tuples.
        
        # CRITICAL CHECK: This was the source of the crash
        if not batch:
            raise IndexError("DataLoader worker returned an empty batch. This means all files "
                             "in the worker's queue (e.g., 9 files) were filtered out by the "
                             "Dataset (e.g., all were too short). Try increasing --num_songs.")

        # Unzip the batch: [(x1, y1), (x2, y2), ...] -> ([x1, x2, ...], [y1, y2, ...])
        x_batch, y_batch = zip(*batch)
        
        # Pad the input sequences (x)
        # We use padding_value=self.pad_token_id
        x_padded = pad_sequence(x_batch, batch_first=True, padding_value=self.pad_token_id)
        
        # Pad the target sequences (y)
        # We use padding_value=-1, which is the standard ignore_index for CrossEntropyLoss
        y_padded = pad_sequence(y_batch, batch_first=True, padding_value=-1)
        
        return x_padded, y_padded

# --- 3. UPDATED get_dataloaders ---

def get_dataloaders(config):

    TOKENIZED_DIR = Path(config.tokenized_dir)
    TOKENIZER_FILE = Path(config.tokenizer_file)
    NUM_SONGS_TO_USE = config.num_songs
    VAL_SPLIT_RATIO = config.val_split
    BATCH_SIZE = config.batch_size
    MAX_SEQ_LEN = config.block_size

    if not TOKENIZER_FILE.exists():
        print(f"Error: Tokenizer file not found at {TOKENIZER_FILE}")
        exit()

    tokenizer = REMI(params=TOKENIZER_FILE)

    # This will be very fast as it's just listing files
    all_json_paths = list(TOKENIZED_DIR.rglob("*.json"))
    print(f"Found {len(all_json_paths)} processed files.")

    if not all_json_paths:
        print(f"Error: No .json files found in {TOKENIZED_DIR}.")
        exit()

    # --- Robustness check for low song count ---
    if NUM_SONGS_TO_USE < BATCH_SIZE and NUM_SONGS_TO_USE < 10:
        print(f"Warning: num_songs ({NUM_SONGS_TO_USE}) is very low. ")
        if NUM_SONGS_TO_USE == 1:
            print("Using num_songs=1. Forcing validation split to 0 and batch_size to 1 to debug.")
            VAL_SPLIT_RATIO = 0
            BATCH_SIZE = 1
        else:
            # If we have 10 songs, BATCH_SIZE should be at most 10
            BATCH_SIZE = max(1, NUM_SONGS_TO_USE // 2)
            print(f"Adjusting batch_size to {BATCH_SIZE} to avoid empty batches.")

    # Shuffle the files
    random.shuffle(all_json_paths)

    # Select the subset
    if NUM_SONGS_TO_USE > len(all_json_paths):
        print(f"Warning: You asked for {NUM_SONGS_TO_USE} songs, but only {len(all_json_paths)} were found.")
        NUM_SONGS_TO_USE = len(all_json_paths)

    subset_paths = all_json_paths[:NUM_SONGS_TO_USE]

    # Create train/val splits
    # Handle the case where VAL_SPLIT_RATIO was forced to 0
    if VAL_SPLIT_RATIO == 0:
        split_index = len(subset_paths)
    else:
        split_index = math.ceil(len(subset_paths) * (1 - VAL_SPLIT_RATIO))
        
    train_paths = subset_paths[:split_index]
    val_paths = subset_paths[split_index:]

    if not train_paths:
        print(f"Error: No files allocated for training. (num_songs: {NUM_SONGS_TO_USE}, val_split: {VAL_SPLIT_RATIO})")
        print("This can happen if num_songs=1 and val_split > 0.")
        sys.exit(1)

    # --- 4. CREATE PYTORCH DATASETS ---

    # Create the Dataset objects
    # *** USE OUR NEW CUSTOM DATASET ***
    train_dataset = MIDITokenDataset(
        files_paths=train_paths,
        max_seq_len=MAX_SEQ_LEN,
    )

    # We must handle the case where val_paths is empty
    val_dataset = None
    if val_paths:
        # *** USE OUR NEW CUSTOM DATASET ***
        val_dataset = MIDITokenDataset(
            files_paths=val_paths,
            max_seq_len=MAX_SEQ_LEN,
        )

    # --- 5. CREATE DATA COLLATOR & DATALOADERS ---

    # *** USE OUR NEW CUSTOM COLLATOR ***
    collator = CustomDataCollator(
        pad_token_id=tokenizer.pad_token_id
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=collator,
        shuffle=True,
        num_workers=4,  # Use multiple workers to load data
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = None # Default to None
    if val_dataset: # Only create a val_loader if the val_dataset exists
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            collate_fn=collator,
            num_workers=4,
            pin_memory=True,
        )

    return train_loader, val_loader