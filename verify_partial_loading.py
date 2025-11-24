import os
import torch
import torch.nn as nn
from types import SimpleNamespace
from trainer import DecoderOnlyTransformer, Trainer, get_optimizer, get_scheduler, get_dataloaders

# Mock Config
class Config:
    # Data
    tokenized_dir = "lmd_matched_processed"
    tokenizer_file = "lmd_matched_tokenizer.json"
    num_songs = 100
    val_split = 0.1
    
    # Model Architecture (Target Model)
    vocab_size = 5000
    block_size = 1024
    n_embed = 512
    n_head = 16       # Target: 16 heads
    n_blocks = 16     # Target: 16 blocks
    dropout = 0.1
    
    # Training
    learning_rate = 3e-4
    max_epochs = 1
    batch_size = 4
    optimizer_type = 'adamw'
    weight_decay = 0.01
    adam_beta1 = 0.9
    adam_beta2 = 0.95
    momentum = 0.9
    grad_clip = 1.0
    
    # Scheduler
    scheduler = 'cosine'
    min_learning_rate = 3e-5
    lr_step_size = 10
    lr_gamma = 0.1
    lr_patience = 3
    
    # Checkpointing
    checkpoint_path = 'trained_models/test_partial_run/model.pt'
    load_checkpoint = False
    run_name = 'test_partial_run'
    compile = False
    
    # Partial Loading Settings
    # We will use a dummy path that we create
    pretrained_path = 'trained_models/dummy_test/model.pt'
    n_layers_to_load = 8
    freeze_loaded = True

config = Config()

def create_dummy_checkpoint(path):
    print(f"Creating dummy checkpoint at {path}...")
    dummy_config = SimpleNamespace(
        vocab_size=5000, n_embed=512, n_head=8, n_blocks=8, block_size=1024, dropout=0.1
    )
    dummy_model = DecoderOnlyTransformer(
        vocab_size=dummy_config.vocab_size,
        n_embed=dummy_config.n_embed,
        n_head=dummy_config.n_head,
        n_blocks=dummy_config.n_blocks,
        block_size=dummy_config.block_size,
        dropout=dummy_config.dropout
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model_state_dict': dummy_model.state_dict(),
        'epoch': 0,
        'best_val_loss': 1.0
    }, path)
    print("Dummy checkpoint created.")

def load_partial_weights(model, checkpoint_path, n_layers_to_load, freeze=False):
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        return
        
    print(f"Loading weights from {checkpoint_path}...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return

    state_dict = checkpoint['model_state_dict']
    
    try:
        model.token_embedding.load_state_dict({'weight': state_dict['token_embedding.weight']})
        model.position_embedding.load_state_dict({'weight': state_dict['position_embedding.weight']})
        print("Loaded embeddings.")
    except Exception as e:
        print(f"Error loading embeddings: {e}")

    if freeze:
        model.token_embedding.weight.requires_grad = False
        model.position_embedding.weight.requires_grad = False
    
    loaded_blocks = 0
    for i in range(n_layers_to_load):
        if i >= len(model.blocks):
            break
            
        block_prefix = f'blocks.{i}.'
        block_state = {}
        for k, v in state_dict.items():
            if k.startswith(block_prefix):
                local_key = k[len(block_prefix):]
                block_state[local_key] = v
        
        if not block_state:
            continue
            
        try:
            model.blocks[i].load_state_dict(block_state)
            loaded_blocks += 1
            
            if freeze:
                for param in model.blocks[i].parameters():
                    param.requires_grad = False
                    
        except Exception as e:
            print(f"Error loading block {i}: {e}")
            
    print(f"Successfully loaded {loaded_blocks} blocks.")
    if freeze:
        print(f"Frozen parameters for embeddings and first {loaded_blocks} blocks.")

# Test Execution
print("Initializing model...")
model = DecoderOnlyTransformer(
    vocab_size=config.vocab_size,
    n_embed=config.n_embed,
    n_head=config.n_head,
    n_blocks=config.n_blocks,
    block_size=config.block_size,
    dropout=config.dropout
)

# Force create dummy checkpoint
create_dummy_checkpoint(config.pretrained_path)

print("Testing partial loading...")
load_partial_weights(model, config.pretrained_path, config.n_layers_to_load, config.freeze_loaded)

# Verify freezing
print("Verifying freezing...")
frozen_ok = True
trainable_ok = True

for name, param in model.named_parameters():
    if 'blocks.0.' in name:
        if param.requires_grad:
            print(f"ERROR: {name} should be frozen but is not.")
            frozen_ok = False
    if 'blocks.15.' in name:
        if not param.requires_grad:
            print(f"ERROR: {name} should be trainable but is not.")
            trainable_ok = False

if frozen_ok and trainable_ok:
    print("Verification SUCCESS: Freezing logic works correctly.")
else:
    print("Verification FAILED: Freezing logic incorrect.")
