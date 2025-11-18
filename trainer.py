#!/usr/bin/env python3
"""
A complete, self-contained script to train a Decoder-Only (GPT-style)
Transformer for sequence generation.

This script includes:
1.  The model definition (`DecoderOnlyTransformer`).
2.  A `Trainer` class to handle training, validation, and checkpointing.
3.  Command-line argument parsing with `argparse` for hyperparameters.
4.  Automatic run directory creation (`trained_models/<run_hash>`) for
    ablation studies, saving config.json, checkpoints, and loss_curves.png.

Example usage:
$ python gpt_model_trainer.py --vocab_size=1024 --n_blocks=8 --n_embed=512 --n_head=8 --batch_size=64 --max_epochs=10 --learning_rate=3e-4 --scheduler='cosine'
"""

import argparse
import math
import os
import time
import tqdm
from typing import Optional, Tuple
import json
import hashlib

import torch
import torch.nn as nn
from torch.nn import functional as F
from dataloader import get_dataloaders
import matplotlib.pyplot as plt

# --- 1. Model Definition ---

class DecoderBlock(nn.Module):
    """A single block of the Transformer decoder."""

    def __init__(self, n_embed: int, n_head: int, block_size: int, dropout: float):
        """
        Args:
            n_embed: The embedding dimension.
            n_head: The number of attention heads.
            block_size: The maximum sequence length (context window).
            dropout: The dropout rate.
        """
        super().__init__()
        head_size = n_embed // n_head
        self.sa = nn.MultiheadAttention(n_embed, n_head, dropout=dropout, batch_first=True)
        self.ffwd = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.GELU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        
        # Causal mask to ensure that attention is only applied to the left
        # in the input sequence.
        # This is persistent, but not a parameter.
        self.register_buffer("mask", torch.triu(torch.ones(block_size, block_size) * float('-inf'), diagonal=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the DecoderBlock.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embed).
            
        Returns:
            Output tensor of shape (batch_size, seq_len, n_embed).
        """
        # Get the sequence length from the input tensor
        B, T, C = x.shape
        
        # Create the causal mask for the current sequence length
        # We can't use the static self.mask directly if T < block_size
        causal_mask = self.mask[:T, :T]

        # Self-attention with residual connection and layer norm
        # Note: MultiheadAttention expects mask: (L, S) or (N*num_heads, L, S)
        # We pass (T, T) which is broadcasted.
        attn_output, _ = self.sa(x, x, x, attn_mask=causal_mask, need_weights=False)
        x = x + self.ln1(attn_output)
        
        # Feed-forward with residual connection and layer norm
        x = x + self.ln2(self.ffwd(x))
        return x


class DecoderOnlyTransformer(nn.Module):
    """
    The full GPT-style Decoder-Only Transformer model.
    """
    def __init__(self, vocab_size: int, n_embed: int, n_head: int, n_blocks: int, block_size: int, dropout: float):
        """
        Args:
            vocab_size: Number of unique tokens in the vocabulary.
            n_embed: The embedding dimension.
            n_head: The number of attention heads.
            n_blocks: The number of decoder blocks.
            block_size: The maximum sequence length (context window).
            dropout: The dropout rate.
        """
        super().__init__()
        self.block_size = block_size
        
        # Token and positional embeddings
        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.position_embedding = nn.Embedding(block_size, n_embed)
        
        # Stack of decoder blocks
        self.blocks = nn.Sequential(*[DecoderBlock(n_embed, n_head, block_size, dropout) for _ in range(n_blocks)])
        
        # Final layer norm and classification head
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

        # Initialize weights
        self.apply(self._init_weights)
        print(f"Model initialized. Total parameters: {self.get_num_params():,}")

    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self) -> int:
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for the model.
        
        Args:
            idx: Input token indices, shape (B, T).
            targets: Target token indices, shape (B, T). Optional.
            
        Returns:
            A tuple containing:
            - logits: Output logits, shape (B, T, vocab_size).
            - loss: Cross-entropy loss (if targets are provided), else None.
        """
        B, T = idx.shape
        if T > self.block_size:
            raise ValueError(f"Input sequence length ({T}) exceeds model block size ({self.block_size})")

        # Get token embeddings
        tok_emb = self.token_embedding(idx) # (B, T, C)
        
        # Get positional embeddings
        # We create positions `0, 1, ..., T-1`
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0) # (1, T)
        pos_emb = self.position_embedding(pos) # (1, T, C)
        
        # Add embeddings
        x = tok_emb + pos_emb # (B, T, C)
        
        # Pass through decoder blocks
        x = self.blocks(x)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Get logits
        logits = self.lm_head(x) # (B, T, vocab_size)

        # Calculate loss if targets are provided
        loss = None
        if targets is not None:
            # Reshape for cross_entropy
            B_logits, T_logits, C_logits = logits.shape
            logits_view = logits.view(B_logits * T_logits, C_logits)
            targets_view = targets.view(B_logits * T_logits)
            loss = F.cross_entropy(logits_view, targets_view, ignore_index=-1) # Assuming -1 is a pad token

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: Optional[int] = None) -> torch.Tensor:
        """
        Generate new token sequences.
        
        Args:
            idx: Current context, shape (B, T).
            max_new_tokens: Number of new tokens to generate.
            temperature: Softmax temperature (lower is sharper, higher is more random).
            top_k: Sample from only the top_k most likely tokens.
            
        Returns:
            Generated sequence, shape (B, T + max_new_tokens).
        """
        self.eval() # Set model to evaluation mode
        for _ in range(max_new_tokens):
            # Crop context if it exceeds block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            
            # Get logits
            logits, _ = self(idx_cond) # (B, T, vocab_size)
            
            # Focus on the last token's logits
            logits = logits[:, -1, :] / temperature # (B, vocab_size)
            
            # Apply Top-K sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Get probabilities
            probs = F.softmax(logits, dim=-1) # (B, vocab_size)
            
            # Sample next token
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            
            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)
            
        self.train() # Set model back to train mode
        return idx

# --- 2. Trainer Class ---

class Trainer:
    """A class to manage the training and evaluation loop."""

    def __init__(self, model, optimizer, scheduler, train_loader, val_loader, config):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Dynamic device selection
        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'
            
        self.model.to(self.device)
        
        # Run-specific directory for checkpoints, config, and plots
        self.run_dir = os.path.dirname(self.config.checkpoint_path)
        os.makedirs(self.run_dir, exist_ok=True)
        
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        
        # History for plotting
        self.train_loss_history = []
        self.val_loss_history = []

        print(f"Trainer initialized. Running on device: {self.device}")
        print(f"Run artifacts will be saved to: {self.run_dir}")

    def _run_epoch(self, split: str) -> float:
        """Run a single epoch of training or validation."""
        is_train = split == 'train'
        self.model.train(is_train)
        
        loader = self.train_loader if is_train else self.val_loader
        if loader is None:
            return float('inf')
            
        total_loss = 0
        num_batches = 0
        
        pbar = enumerate(loader)
        for it, (x, y) in tqdm.tqdm(pbar):
            x, y = x.to(self.device), y.to(self.device)
            
            with torch.set_grad_enabled(is_train):
                logits, loss = self.model(x, y)
                
            if loss is None:
                continue

            if is_train:
                self.optimizer.zero_grad()
                loss.backward()
                # Gradient clipping
                if self.config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.optimizer.step()
                
            total_loss += loss.item()
            num_batches += 1

        if num_batches == 0:
            return float('inf')
            
        avg_loss = total_loss / num_batches
        return avg_loss

    def _plot_loss_curves(self):
        """Plots and saves the training and validation loss curves."""
        if not self.train_loss_history and not self.val_loss_history:
            print("No loss history to plot.")
            return

        epochs_ran = len(self.train_loss_history)
        if epochs_ran == 0:
            print("No epochs were run, skipping plot.")
            return
            
        epoch_axis = range(self.start_epoch, self.start_epoch + epochs_ran)
        
        plt.figure(figsize=(12, 6))
        
        # Plot train loss
        plt.plot(epoch_axis, self.train_loss_history, 'b-', label='Train Loss')
        # Plot validation loss
        plt.plot(epoch_axis, self.val_loss_history, '-', color='orange', label='Validation Loss')
        
        # Add red dots for checkpoints (saved every epoch)
        plt.plot(epoch_axis, self.train_loss_history, 'ro', markersize=4, label='Checkpoint (Epoch)')
        plt.plot(epoch_axis, self.val_loss_history, 'ro', markersize=4, label='_nolegend_') # Avoid duplicate legend
        
        plt.title('Training & Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        save_path = os.path.join(self.run_dir, 'loss_curves.png')
        try:
            plt.savefig(save_path)
            print(f"Saved loss curves to {save_path}")
        except Exception as e:
            print(f"Error saving loss curves: {e}")
        plt.close()


    def save_checkpoint(self, epoch: int, is_best: bool):
        """Save a checkpoint."""
        if not self.config.checkpoint_path:
            return
            
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }
        
        # Save last checkpoint (e.g., trained_models/<hash>/model.pt)
        torch.save(state, self.config.checkpoint_path)
        print(f"Saved checkpoint #{epoch} to {self.config.checkpoint_path}")

        # Save best checkpoint (e.g., trained_models/<hash>/model_best.pt)
        if is_best:
            best_path = self.config.checkpoint_path.replace('.pt', '_best.pt')
            torch.save(state, best_path)
            print(f"Saved *best* checkpoint to {best_path}")

    def load_checkpoint(self):
        """Load a checkpoint."""
        if not self.config.load_checkpoint or not self.config.checkpoint_path or not os.path.exists(self.config.checkpoint_path):
            print("No checkpoint found to load.")
            return

        try:
            checkpoint = torch.load(self.config.checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if self.scheduler and checkpoint.get('scheduler_state_dict'):
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_val_loss = checkpoint['best_val_loss']
            
            print(f"Loaded checkpoint from {self.config.checkpoint_path}. Resuming from epoch {self.start_epoch}.")
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting from scratch.")
            self.start_epoch = 0
            self.best_val_loss = float('inf')

    def train(self):
        """Run the full training loop."""
        
        # Save config.json to the run directory
        config_path = os.path.join(self.run_dir, 'config.json')
        try:
            with open(config_path, 'w') as f:
                # Convert namespace to dict for json serialization
                json.dump(vars(self.config), f, indent=4)
            print(f"Saved config to {config_path}")
        except Exception as e:
            print(f"Error saving config.json: {e}")

        # Load checkpoint if requested
        self.load_checkpoint()

        for epoch in tqdm.tqdm(range(self.start_epoch, self.config.max_epochs)):
            epoch_start_time = time.time()
            
            train_loss = self._run_epoch('train')
            val_loss = self._run_epoch('val')
            
            # Store loss history
            self.train_loss_history.append(train_loss)
            self.val_loss_history.append(val_loss)
            
            if self.scheduler:
                # Scheduler step can be based on val_loss or epoch
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            epoch_duration = time.time() - epoch_start_time
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f} | Time: {epoch_duration:.2f}s")
            
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            self.save_checkpoint(epoch, is_best)
            
        print(f"Training finished. Best validation loss: {self.best_val_loss:.4f}")
        
        # Plot and save loss curves
        self._plot_loss_curves()


# --- 4. Main Function ---

def get_config_hash(config: argparse.Namespace) -> str:
    """Creates a unique hash from the model and training config."""
    # Select parameters that define the run
    hash_params = {
        'vocab_size': config.vocab_size,
        'block_size': config.block_size,
        'n_embed': config.n_embed,
        'n_blocks': config.n_blocks,
        'n_head': config.n_head,
        'dropout': config.dropout,
        'learning_rate': config.learning_rate,
        'batch_size': config.batch_size,
        'optimizer_type': config.optimizer_type,
        'weight_decay': config.weight_decay,
        'adam_beta1': config.adam_beta1,
        'adam_beta2': config.adam_beta2,
        'momentum': config.momentum,
        'scheduler': config.scheduler,
        'min_learning_rate': config.min_learning_rate,
        'lr_step_size': config.lr_step_size,
        'lr_gamma': config.lr_gamma,
        'lr_patience': config.lr_patience,
    }
    
    # Sort keys for consistent hashing
    sorted_params = json.dumps(hash_params, sort_keys=True)
    
    # Create hash
    h = hashlib.md5(sorted_params.encode('utf-8')).hexdigest()
    
    # Return a short hash
    return h[:10] # Using first 10 chars

def get_optimizer(model, config):
    """Create and return the optimizer."""
    # Start with all parameters
    params = model.parameters()
    
    # You could implement weight decay exclusion for bias/LayerNorm here
    
    if config.optimizer_type == 'adamw':
        print("Using AdamW optimizer")
        optimizer = torch.optim.AdamW(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(config.adam_beta1, config.adam_beta2)
        )
    elif config.optimizer_type == 'sgd':
        print("Using SGD optimizer")
        optimizer = torch.optim.SGD(
            params,
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer type: {config.optimizer_type}")
        
    return optimizer

def get_scheduler(optimizer, config, total_steps):
    """Create and return the learning rate scheduler."""
    if config.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=config.min_learning_rate
        )
    elif config.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.lr_step_size,
            gamma=config.lr_gamma
        )
    elif config.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.lr_gamma,
            patience=config.lr_patience
        )
    elif config.scheduler == 'exponential':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=config.lr_gamma
        )
    elif config.scheduler == 'none':
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler type: {config.scheduler}")
        
    return scheduler

def main():
    parser = argparse.ArgumentParser(description="Train a GPT-style Transformer for Musical Autocomplete")

    # --- Training hyperparameters ---
    parser.add_argument('--tokenized_dir', type=str, default="lmd_matched_processed", help="The directory of tokenized songs.")
    parser.add_argument('--tokenizer_file', type=str, default="lmd_matched_tokenizer.json", help="The tokenizer to use for tokenizing/detokenizing songs")
    parser.add_argument('--num_songs', type=int, default=1000, help="Number of songs to use for training")
    parser.add_argument('--val_split', type=float, default=0.1, help="Percentage of songs to use for validation: (0, 1)")

    # --- Model Hyperparameters ---
    parser.add_argument('--vocab_size', type=int, default=5000, help="Vocabulary size.")
    parser.add_argument('--block_size', type=int, default=1024, help="Max sequence length (context window).")
    parser.add_argument('--n_embed', type=int, default=512, help="Embedding dimension.")
    parser.add_argument('--n_blocks', type=int, default=8, help="Number of Transformer blocks.")
    parser.add_argument('--n_head', type=int, default=8, help="Number of attention heads.")
    parser.add_argument('--dropout', type=float, default=0.1, help="Dropout rate.")

    # --- Training Hyperparameters ---
    parser.add_argument('--learning_rate', type=float, default=3e-4, help="Peak learning rate.")
    parser.add_argument('--max_epochs', type=int, default=10, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size.")
    parser.add_argument('--optimizer_type', type=str, default='adamw', choices=['adamw', 'sgd'], help="Optimizer type.")
    parser.add_argument('--weight_decay', type=float, default=0.01, help="Weight decay (AdamW) or L2 penalty (SGD).")
    parser.add_argument('--adam_beta1', type=float, default=0.9, help="Adam optimizer beta1.")
    parser.add_argument('--adam_beta2', type=float, default=0.95, help="Adam optimizer beta2.")
    parser.add_argument('--momentum', type=float, default=0.9, help="Momentum for SGD optimizer.")
    parser.add_argument('--grad_clip', type=float, default=1.0, help="Gradient clipping value (0 for no clipping).")

    # --- Scheduler ---
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'step', 'plateau', 'exponential', 'none'], help="LR scheduler type.")
    parser.add_argument('--min_learning_rate', type=float, default=3e-5, help="Minimum LR for cosine scheduler.")
    parser.add_argument('--lr_step_size', type=int, default=10, help="Step size for 'step' scheduler.")
    parser.add_argument('--lr_gamma', type=float, default=0.1, help="LR decay factor for 'step', 'plateau', or 'exponential' schedulers.")
    parser.add_argument('--lr_patience', type=int, default=3, help="Patience for 'plateau' scheduler.")

    # --- Checkpointing & Run Naming ---
    parser.add_argument('--checkpoint_path', type=str, default='trained_models/model.pt', help="Path to save/load checkpoints. Base directory will be 'trained_models'.")
    parser.add_argument('--load_checkpoint', action='store_true', help="Flag to load a checkpoint if it exists.")
    parser.add_argument('--run_name', type=str, default=None, help="A custom name for the run directory. If None, a hash is used.")
    
    parser.add_argument("--compile", action="store_true", help="Use torch.compile() to optimize the model (requires PyTorch 2.0+)")

    config = parser.parse_args()

    # --- Setup ---
    torch.manual_seed(42)

    # Create unique run hash and checkpoint directory
    run_hash = get_config_hash(config)
    
    # Use custom run_name if provided, otherwise fall back to hash
    run_dir_name = config.run_name if config.run_name else run_hash
    print(f"Run name: {run_dir_name}")
    
    # Update checkpoint_path to be unique per run
    # e.g., 'trained_models/model.pt' -> 'trained_models/<run_dir_name>/model.pt'
    base_dir = os.path.dirname(config.checkpoint_path) # e.g., 'trained_models'
    filename = os.path.basename(config.checkpoint_path) # e.g., 'model.pt'
    config.checkpoint_path = os.path.join(base_dir, run_dir_name, filename)
    # the Trainer class will create this directory

    train_loader, val_loader = get_dataloaders(config)
    
    # Calculate total steps for cosine scheduler
    total_steps = len(train_loader) * config.max_epochs

    # 2. Initialize Model
    model = DecoderOnlyTransformer(
        vocab_size=config.vocab_size,
        n_embed=config.n_embed,
        n_head=config.n_head,
        n_blocks=config.n_blocks,
        block_size=config.block_size,
        dropout=config.dropout
    )

    if config.compile:
        if hasattr(torch, 'compile'):
            print("Compiling the model... (this may take a moment)")
            try:
                model = torch.compile(model)
                print("Model compiled successfully.")
            except Exception as e:
                print(f"Model compilation failed: {e}. Running un-compiled.")
        else:
            print("torch.compile not found. Running un-compiled. (Requires PyTorch 2.0+)")

    # 3. Initialize Optimizer and Scheduler
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config, total_steps)

    # 4. Initialize Trainer and Start Training
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )
    
    trainer.train()


if __name__ == "__main__":
    main()
