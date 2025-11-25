#!/usr/bin/env python3
"""
A complete, self-contained script for running inference (generation)
from a trained Decoder-Only Transformer model checkpoint.

The script loads the model configuration and weights from a specified
run directory and generates a sequence based on a user-provided prompt.
The output is saved as a placeholder MIDI file.

The model architecture is now imported directly from trainer.py to maintain
a single source of truth for the definition.

Example usage (assuming a run hash 'abcdef1234' exists):
$ python inference.py --run_dir trained_models/abcdef1234 --prompt_tokens 1,2,3,4,5 --max_new_tokens 512 --soundfont_path /path/to/your/soundfont.sf2
"""

import argparse
import os
import json
import time
from typing import Optional, Tuple
import subprocess
import shutil

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import tqdm
import miditok
from pathlib import Path

# --- 1. Import Model Definitions from trainer.py ---
# Note: DecoderBlock is also imported as it is defined at the top level in trainer.py
from trainer import DecoderOnlyTransformer, DecoderBlock 

# --- 2. Token to MIDI Converter (Real Implementation) ---

def save_tokens_as_midi(token_ids: np.ndarray, tokenizer_path: str, output_path: str):
    """
    Converts a sequence of token IDs into a MIDI file using a miditok tokenizer.
    
    Args:
        token_ids: 1D numpy array of integer token IDs.
        tokenizer_path: Path to the trained miditok tokenizer JSON file.
        output_path: The full path to save the resulting .mid file.
    """
    
    # --- V2 API Implementation ---
    
    # 1. Load the tokenizer config
    try:
        # --- THE FIX ---
        # We must load the *entire* tokenizer (config + vocab)
        # using the .load() classmethod, not just the config.
        # This will load the ~5000 tokens your model was trained on.
        tokenizer = miditok.REMI(params=Path(tokenizer_path))

    except FileNotFoundError:
        raise FileNotFoundError(f"Tokenizer file not found at: {tokenizer_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load tokenizer from file: {e}. Ensure this is a valid tokenizer JSON file saved with miditok V2's .save() method.")

    # 2. Ensure token_ids is a 1D list of standard Python integers
    if token_ids.ndim > 1:
        token_ids = token_ids.flatten()
    token_ids_list = token_ids.tolist()

    # 3. Convert token IDs directly to a MIDI object (V2 API)
    try:
        # The V2 API cleanly handles converting IDs back to a MidiFile object
        midi_object = tokenizer([token_ids_list])
        
    except IndexError as e:
        raise ValueError(f"A generated token ID is out of the tokenizer's vocabulary range: {e}. This may happen if the wrong vocab_size was used during training or inference.")
    except Exception as e:
        raise RuntimeError(f"Failed during token ID to MIDI conversion: {e}. The generated sequence may be malformed or incompatible with the tokenizer.")

    # 4. Save the MIDI file
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # .dump() is the method for MidiFile objects
        midi_object.dump_midi(output_path)
        print(f"\\n[SUCCESS] Saved generated MIDI file to: {output_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to save MIDI file: {e}")


# --- 3. MIDI to MP3 Converter ---

def convert_midi_to_mp3(midi_path: str, soundfont_path: str):
    """
    Converts a MIDI file to an MP3 file using FluidSynth and FFmpeg.
    
    This function requires `fluidsynth` and `ffmpeg` to be installed
    and available in the system's PATH.
    
    Args:
        midi_path: Path to the input .mid file.
        soundfont_path: Path to the .sf2 SoundFont file.
    """
    
    # 1. Check for required external dependencies
    if shutil.which("fluidsynth") is None:
        print("[WARNING] `fluidsynth` binary not found. Cannot synthesize MIDI to MP3.")
        print("           Please install FluidSynth (e.g., 'sudo apt-get install fluidsynth').")
        return
        
    if shutil.which("ffmpeg") is None:
        print("[WARNING] `ffmpeg` binary not found. Cannot convert WAV to MP3.")
        print("           Please install FFmpeg (e.g., 'sudo apt-get install ffmpeg').")
        return

    # 2. Check if SoundFont file exists
    if not os.path.exists(soundfont_path):
        print(f"[WARNING] SoundFont file not found at: {soundfont_path}")
        print("           Skipping MP3 conversion.")
        return

    print(f"[INFO] Synthesizing MIDI to MP3 using SoundFont: {soundfont_path}")
    
    # Define file paths
    wav_path = midi_path.replace('.mid', '.wav')
    mp3_path = midi_path.replace('.mid', '.mp3')

    try:
        # 3. Step 1: Synthesize MIDI to WAV using FluidSynth
        # -ni: Non-interactive
        # -F: Output to file
        # -r 44100: Sample rate
        fluidsynth_cmd = [
            'fluidsynth',
            '-ni',
            soundfont_path,
            midi_path,
            '-F', wav_path,
            '-r', '44100'
        ]
        subprocess.run(fluidsynth_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        print(f"[INFO] Successfully created intermediate WAV: {wav_path}")

        # 4. Step 2: Convert WAV to MP3 using FFmpeg
        # -i: Input file
        # -b:a 192k: Audio bitrate (192 kbps)
        # -y: Overwrite output file if it exists
        ffmpeg_cmd = [
            'ffmpeg',
            '-i', wav_path,
            '-b:a', '192k',
            '-y',  # Overwrite output file
            mp3_path
        ]
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        print(f"[SUCCESS] Saved generated MP3 file to: {mp3_path}")

    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed during audio conversion. Return code: {e.returncode}")
        if e.stderr:
            print(f"         Error output:\\n{e.stderr.decode('utf-8')}")
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred during conversion: {e}")
    finally:
        # 5. Step 3: Clean up intermediate WAV file
        if os.path.exists(wav_path):
            os.remove(wav_path)


# --- 4. Main Inference Function ---

def run_inference(config: argparse.Namespace):
    """Loads model, generates sequence, and saves the output."""
    
    run_dir = config.run_dir
    checkpoint_file = os.path.join(run_dir, 'model.pt')
    config_file = os.path.join(run_dir, 'config.json')

    # 1. Load Configuration
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
    with open(config_file, 'r') as f:
        config_data = json.load(f)
        
    # Overwrite relevant config fields with CLI arguments if provided
    for key, value in config_data.items():
        if not hasattr(config, key) or getattr(config, key) is None:
            setattr(config, key, value)
    
    # Ensure mandatory config for model are present
    if not all(hasattr(config, k) for k in ['vocab_size', 'n_embed', 'n_head', 'n_blocks', 'block_size', 'dropout']):
        raise ValueError("Missing essential model configuration parameters.")

    # 1.a. Check for tokenizer file
    if not os.path.exists(config.tokenizer_file):
        raise FileNotFoundError(f"Tokenizer file not found at path: {config.tokenizer_file}. Please provide the correct path via --tokenizer_file")

    # 2. Setup Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running inference on device: {device}")
    
    # 3. Initialize Model (using the imported class)
    model = DecoderOnlyTransformer(
        vocab_size=config.vocab_size,
        n_embed=config.n_embed,
        n_head=config.n_head,
        n_blocks=config.n_blocks,
        block_size=config.block_size,
        dropout=config.dropout
    )
    
    # 4. Load Checkpoint Weights
    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")

    # Check if the file is a Git LFS pointer
    try:
        with open(checkpoint_file, 'rb') as f:
            header = f.read(100)
            if header.startswith(b"version https://git-lfs"):
                raise RuntimeError(f"The model file at {checkpoint_file} appears to be a Git LFS pointer, not the actual binary. Please run 'git lfs pull' to download the actual model weights.")
    except Exception as e:
        print(f"[WARNING] Could not check file header: {e}")

    print(f"Loading weights from {checkpoint_file}...")
    checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    # 5. Prepare Prompt
    try:
        # Assuming prompt is a comma-separated list of token indices (integers)
        prompt_tokens_list = [int(t.strip()) for t in config.prompt_tokens.split(',')]
        if not prompt_tokens_list:
             raise ValueError("Prompt tokens list is empty after parsing.")
    except ValueError:
        raise ValueError("Invalid prompt tokens. Ensure they are comma-separated integers (e.g., '10,20,30').")
        
    # Convert to tensor: (1, seq_len)
    prompt_tensor = torch.tensor([prompt_tokens_list], dtype=torch.long, device=device)
    print(f"\\n[INFO] Input prompt size: {prompt_tensor.size(1)} tokens.")

    # 6. Generate Sequence
    print(f"[INFO] Generating {config.max_new_tokens} new tokens...")
    start_time = time.time()
    
    # Since the generate method is on the DecoderOnlyTransformer class (now imported), 
    # the call remains the same.
    generated_sequence = model.generate(
        idx=prompt_tensor,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_k=config.top_k
    )
    
    generation_time = time.time() - start_time
    total_tokens = generated_sequence.size(1)
    
    print(f"[INFO] Generation finished in {generation_time:.2f}s.")
    print(f"[INFO] Total sequence length: {total_tokens} tokens.")
    
    # Convert back to numpy array on CPU
    output_tokens_np = generated_sequence[0].cpu().numpy()

    # 7. Save Output File
    
    # Create the structured output path: inference/run_directory_name/output_<timestamp>.mid
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = os.path.basename(run_dir.strip(os.sep))
    inference_dir = os.path.join('inference', run_name)
    output_filename = f"generated_{timestamp}_{total_tokens}tokens.mid"
    output_path = os.path.join(inference_dir, output_filename)
    
    # Call the new MIDI conversion function
    save_tokens_as_midi(
        token_ids=output_tokens_np,
        tokenizer_path=config.tokenizer_file,
        output_path=output_path
    )
    
    return output_path

# Alias for API usage
run_inference_api = run_inference

# --- 5. Argparse Setup ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a trained GPT-style Transformer model.")

    parser.add_argument('--run_dir', type=str, required=True, help="Path to the training run directory (e.g., 'trained_models/abcdef1234').")
    parser.add_argument('--prompt_tokens', type=str, required=True, help="Initial sequence of tokens (comma-separated integers, e.g., '10,20,30').")
    parser.add_argument('--max_new_tokens', type=int, default=512, help="Number of new tokens to generate.")
    parser.add_argument('--temperature', type=float, default=1.2, help="Softmax temperature for sampling.")
    parser.add_argument('--top_k', type=int, default=50, help="Sample from the top K most likely tokens.")
    
    # Un-suppress the tokenizer_file argument
    parser.add_argument('--tokenizer_file', type=str, default="lmd_matched_tokenizer.json", help="Path to the trained miditok tokenizer JSON file.")

    config = parser.parse_args()
    
    run_inference(config)