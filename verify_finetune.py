import os
import torch
import shutil
from pathlib import Path
from miditok import REMI
from trainer import DecoderOnlyTransformer, Trainer, get_optimizer, get_scheduler, get_dataloaders
from types import SimpleNamespace

# Mock Config
class Config:
    # --- New Data Settings ---
    raw_data_dir = "test_hq_raw"
    processed_data_dir = "test_hq_processed"
    tokenizer_file = "lmd_matched_tokenizer.json"
    
    # --- Fine-tuning Settings ---
    pretrained_model_path = "trained_models/test_finetune_pretrained/model.pt"
    run_name = "test_finetune_run"
    
    # --- Training Hyperparameters ---
    learning_rate = 1e-5 
    max_epochs = 1
    batch_size = 2
    
    # --- Model Architecture ---
    vocab_size = 5000
    block_size = 1024
    n_embed = 512
    n_head = 8
    n_blocks = 8
    dropout = 0.1
    
    # --- Other Settings ---
    num_songs = 10 
    val_split = 0.1
    optimizer_type = 'adamw'
    weight_decay = 0.01
    adam_beta1 = 0.9
    adam_beta2 = 0.95
    momentum = 0.9
    grad_clip = 1.0
    scheduler = 'cosine'
    min_learning_rate = 1e-6
    lr_step_size = 10
    lr_gamma = 0.1
    lr_patience = 3
    checkpoint_path = f'trained_models/{run_name}/model.pt'
    load_checkpoint = False
    compile = False
    
    @property
    def tokenized_dir(self):
        return self.processed_data_dir

config = Config()

def create_dummy_data(config):
    print("Creating dummy data...")
    os.makedirs(config.raw_data_dir, exist_ok=True)
    os.makedirs(config.processed_data_dir, exist_ok=True)
    
    # Create a dummy tokenized file directly to bypass tokenizer requirement for valid MIDI
    # We simulate the output of the tokenizer
    import json
    dummy_tokens = {"ids": [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 100]} # Long enough sequence
    
    with open(os.path.join(config.processed_data_dir, "dummy_1.json"), "w") as f:
        json.dump(dummy_tokens, f)
    with open(os.path.join(config.processed_data_dir, "dummy_2.json"), "w") as f:
        json.dump(dummy_tokens, f)
        
    print("Dummy processed data created.")

def create_dummy_pretrained_model(config):
    print("Creating dummy pretrained model...")
    os.makedirs(os.path.dirname(config.pretrained_model_path), exist_ok=True)
    
    model = DecoderOnlyTransformer(
        vocab_size=config.vocab_size,
        n_embed=config.n_embed,
        n_head=config.n_head,
        n_blocks=config.n_blocks,
        block_size=config.block_size,
        dropout=config.dropout
    )
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': 0,
        'best_val_loss': 10.0
    }, config.pretrained_model_path)
    print("Dummy pretrained model created.")

def cleanup(config):
    print("Cleaning up...")
    shutil.rmtree(config.raw_data_dir, ignore_errors=True)
    shutil.rmtree(config.processed_data_dir, ignore_errors=True)
    shutil.rmtree(os.path.dirname(config.pretrained_model_path), ignore_errors=True)
    shutil.rmtree(os.path.dirname(config.checkpoint_path), ignore_errors=True)

def verify():
    create_dummy_data(config)
    create_dummy_pretrained_model(config)
    
    print("Initializing model...")
    model = DecoderOnlyTransformer(
        vocab_size=config.vocab_size,
        n_embed=config.n_embed,
        n_head=config.n_head,
        n_blocks=config.n_blocks,
        block_size=config.block_size,
        dropout=config.dropout
    )
    
    print("Loading weights...")
    checkpoint = torch.load(config.pretrained_model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Weights loaded.")
    
    print("Setting up training...")
    # Mock tokenizer file existence for get_dataloaders
    if not os.path.exists(config.tokenizer_file):
        # Create a dummy tokenizer file if it doesn't exist
        print("Creating dummy tokenizer file for test...")
        from miditok import REMI
        tokenizer = REMI()
        tokenizer.save(config.tokenizer_file)

    train_loader, val_loader = get_dataloaders(config)
    
    total_steps = len(train_loader) * config.max_epochs
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config, total_steps)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )
    
    print("Running 1 epoch of training...")
    trainer.train()
    print("Verification SUCCESS.")
    
    cleanup(config)

if __name__ == "__main__":
    verify()
