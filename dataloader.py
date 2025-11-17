import torch
from torch.utils.data import DataLoader
from miditok import REMI
from miditok.pytorch_data import DatasetMIDI, DataCollator
from pathlib import Path
import random
import math
import sys

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
    train_dataset = DatasetMIDI(
        tokenizer=tokenizer,
        files_paths=train_paths,
        max_seq_len=MAX_SEQ_LEN,
        bos_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer["BOS_None"],
    )

    # We must handle the case where val_paths is empty
    val_dataset = None
    if val_paths:
        val_dataset = DatasetMIDI(
            tokenizer=tokenizer,
            files_paths=val_paths,
            max_seq_len=MAX_SEQ_LEN,
            bos_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer["BOS_None"],
        )

    # --- 5. CREATE DATA COLLATOR & DATALOADERS ---

    # The collator handles padding batches
    # This is essential as sequences will have different lengths
    collator = DataCollator(
        pad_token_id=tokenizer.pad_token_id,
        copy_inputs_as_labels=True  # For autoregressive training
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