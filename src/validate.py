########################################################################################################
# RWKV Checkpoint Validation Script - Modified from https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import os
import re
import glob
import json
import logging
import torch
import datetime
import numpy as np
from pathlib import Path
from tqdm import tqdm
from omegaconf import OmegaConf
import hydra
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.utilities import rank_zero_info
from argparse import Namespace
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import deepspeed

# Configure logging
logging.basicConfig(level=logging.INFO)

@hydra.main(config_path="configs", config_name="config")
def main(config):
    # Set a fixed random seed for replicability
    FIXED_SEED = 42
    pl.seed_everything(FIXED_SEED)
    
    # Disable wandb
    os.environ["WANDB_MODE"] = "disabled"
    
    # Create results directory
    results_dir = os.path.join(config.proj_dir, "validation_results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Setup logging file
    timestamp = datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
    log_file = os.path.join(results_dir, f"validation_results_{timestamp}.csv")
    
    # Initialize CSV file with header
    with open(log_file, 'w') as f:
        f.write("checkpoint,validation_loss\n")
    
    # Configure trainer arguments
    config.trainer.devices = torch.cuda.device_count()
    args = Namespace(**OmegaConf.to_container(config.trainer, resolve=True))
    
    # Set trainer configurations for validation only
    args.enable_checkpointing = False
    args.replace_sampler_ddp = False
    args.logger = False
    args.num_sanity_val_steps = 0
    args.log_every_n_steps = int(1e20)
    
    # Setting environment variables and configurations
    config.betas = (config.training.beta1, config.training.beta2)
    config.real_bsz = int(args.num_nodes) * int(args.devices) * config.training.micro_bsz
    os.environ["RWKV_CTXLEN"] = str(config.model.ctx_len)
    os.environ["RWKV_HEAD_SIZE_A"] = str(config.model.head_size_a)
    os.environ["RWKV_CHUNK_LEN"] = str(config.model.chunk_len)
    
    # Set model dimensions if not provided
    if config.model.dim_att <= 0:
        config.model.dim_att = config.model.n_embd
    if config.model.dim_ffn <= 0:
        config.model.dim_ffn = int((config.model.n_embd * 3.5) // 32 * 32)
    
    # Configure torch precision
    assert config.trainer.precision in ["tf32", "16", "bf16"]
    os.environ["RWKV_FLOAT_MODE"] = config.trainer.precision
    
    # Setup CUDA settings
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    
    # Convert precision format
    if "32" in config.trainer.precision:
        config.trainer.precision = 32
    elif config.trainer.precision == "16":
        config.trainer.precision = 16
    else:
        config.trainer.precision = "bf16"
    
    # Set data source directory
    os.environ["RWKV_SRC_DIR"] = config.data.src_dir
    
    # Import required modules
    from model import RWKV
    from dataset import MIDIDataset, DataCollatorNoneFilter
    from datasets import load_from_disk
    from miditok import MMM
    
    # Setup tokenizer
    dc = config.data
    tokenizer = MMM(params=dc.tokenizer_path)
    config.model.vocab_size = tokenizer.vocab_size
    
    # Set data configuration
    dc.tracks_selection_random_ratio_range = (0.4, 1)
    dc.ratios_range_bar_infilling_duration = (0.1, 0.4)
    dc.acs_random_ratio_range = (0.05, 0.9)
    dc.tracks_idx_random_ratio_range = (0.1, 1)
    dc.bars_idx_random_ratio_range = (0.1, 0.7)
    dc.data_augmentation_offsets = (6, 2, 0)
    
    # Load dataset
    ds = load_from_disk(dc.prefiltered_dataset_path)
    
    # Create validation dataset only
    val_data = MIDIDataset(
        ds["validation"],
        tokenizer,
        dc.max_seq_len,
        dc.tracks_selection_random_ratio_range,
        dc.data_augmentation_offsets,
        dc.ratio_bar_infilling,
        dc.ratios_range_bar_infilling_duration,
        ac_random_ratio_range=dc.acs_random_ratio_range,
        ac_tracks_random_ratio_range=dc.tracks_idx_random_ratio_range,
        ac_bars_random_ratio_range=dc.bars_idx_random_ratio_range
    )
    
    # Setup data collator
    collator = DataCollatorNoneFilter(pad_token_id=tokenizer.pad_token_id, max_length=config.model.ctx_len)
    
    # Get all checkpoint files matching the pattern rwkv-X.pth where X is a number
    checkpoint_dir = "/home/christian/RWKV-LM/RWKV-v5/out/L8-D512-x070"
    checkpoint_pattern = os.path.join(checkpoint_dir, "rwkv-*.pth")
    checkpoint_files = glob.glob(checkpoint_pattern)
    
    # Filter to only include checkpoints with format rwkv-X.pth where X is a number
    checkpoint_files = [cp for cp in checkpoint_files if re.match(r'.*rwkv-\d{1,2}\.pth$', cp)]
    
    # Sort checkpoints by number
    checkpoint_files.sort(key=lambda x: int(re.search(r'rwkv-(\d{1,2})\.pth$', x).group(1)))
    
    if not checkpoint_files:
        rank_zero_info(f"No checkpoint files found matching pattern {checkpoint_pattern}")
        return
    
    rank_zero_info(f"Found {len(checkpoint_files)} checkpoints: {[os.path.basename(cp) for cp in checkpoint_files]}")
    
    # Process each checkpoint
    for checkpoint_path in checkpoint_files:
        checkpoint_name = os.path.basename(checkpoint_path)
        rank_zero_info(f"Validating checkpoint: {checkpoint_name}")
        
        # Create model
        model = RWKV(config)
        
        # Load checkpoint
        try:
            load_dict = torch.load(checkpoint_path, map_location="cpu")
            load_keys = list(load_dict.keys())
            for k in load_keys:
                if k.startswith('_forward_module.'):
                    load_dict[k.replace('_forward_module.', '')] = load_dict[k]
                    del load_dict[k]
            model.load_state_dict(load_dict)
        except Exception as e:
            rank_zero_info(f"Error loading checkpoint {checkpoint_path}: {e}")
            continue
        
        # Setup validation data loader with fixed seed
        val_sampler = DistributedSampler(
            val_data,
            num_replicas=1,  # Single process validation
            rank=0,
            shuffle=False,
            seed=FIXED_SEED
        )
        
        val_loader = DataLoader(
            val_data, 
            sampler=val_sampler, 
            pin_memory=True, 
            batch_size=config.training.micro_bsz, 
            num_workers=config.training.dataloader_num_workers, 
            persistent_workers=False, 
            drop_last=True, 
            collate_fn=collator
        )
        
        # Setup trainer for validation only
        trainer = Trainer.from_argparse_args(
            args,
            max_epochs=0,  # No training, only validation
            limit_train_batches=0,  # No training
            limit_val_batches=1.0,  # Full validation
        )
        
        # Run validation
        val_results = trainer.validate(model, val_loader)
        
        # Extract validation loss
        val_loss = val_results[0]['val_loss']
        
        # Log results
        rank_zero_info(f"Checkpoint {checkpoint_name} - Validation Loss: {val_loss:.6f}")
        
        # Save to CSV file
        with open(log_file, 'a') as f:
            f.write(f"{checkpoint_name},{val_loss:.6f}\n")
    
    # Generate final summary report
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        
        # Read results
        results_df = pd.read_csv(log_file)
        
        # Extract numbers from checkpoint names
        results_df['epoch'] = results_df['checkpoint'].apply(
            lambda x: int(re.search(r'rwkv-(\d{1,2})\.pth$', x).group(1))
        )
        
        # Sort by epoch
        results_df = results_df.sort_values('epoch')
        
        # Find best checkpoint
        best_idx = results_df['validation_loss'].idxmin()
        best_checkpoint = results_df.loc[best_idx, 'checkpoint']
        best_loss = results_df.loc[best_idx, 'validation_loss']
        
        # Create summary file
        summary_file = os.path.join(results_dir, f"validation_summary_{timestamp}.txt")
        with open(summary_file, 'w') as f:
            f.write(f"Validation Summary - {timestamp}\n")
            f.write(f"Total Checkpoints Evaluated: {len(results_df)}\n")
            f.write(f"Best Checkpoint: {best_checkpoint} (Loss: {best_loss:.6f})\n\n")
            f.write("All Results (sorted by validation loss):\n")
            for _, row in results_df.sort_values('validation_loss').iterrows():
                f.write(f"  {row['checkpoint']}: {row['validation_loss']:.6f}\n")
        
        # Create a plot of validation loss vs epoch
        plt.figure(figsize=(10, 6))
        plt.plot(results_df['epoch'], results_df['validation_loss'], marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Validation Loss')
        plt.title('RWKV Model Validation Loss Across Checkpoints')
        plt.grid(True)
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(results_dir, f"validation_plot_{timestamp}.png")
        plt.savefig(plot_file)
        
        rank_zero_info(f"\nValidation complete. Results saved to {log_file}")
        rank_zero_info(f"Summary saved to {summary_file}")
        rank_zero_info(f"Plot saved to {plot_file}")
        rank_zero_info(f"\nBest checkpoint: {best_checkpoint} with validation loss: {best_loss:.6f}")
        
    except Exception as e:
        rank_zero_info(f"Error generating summary: {e}")
        rank_zero_info(f"Raw results saved to {log_file}")

if __name__ == "__main__":
    main()