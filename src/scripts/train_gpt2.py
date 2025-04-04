import torch
from torch.utils.data import DataLoader
from transformers import GPT2Config, AutoModelForCausalLM
from tqdm import tqdm
import logging
import os
import numpy as np
import hydra
from omegaconf import OmegaConf
import time
import wandb
from miditok.pytorch_data import DataCollator
from torch.cuda.amp import autocast, GradScaler


class DataCollatorNoneFilter:
    def __init__(self, pad_token_id=0, max_length=2048):
        self.pad_token_id = pad_token_id
        self.max_length = max_length
        self.collator = DataCollator(pad_token_id)
    
    def __call__(self, batch):
        collated_batch = self.collator(batch)
        
        # Get the current sequence length
        input_ids = collated_batch["input_ids"]
        labels = collated_batch["labels"]
        batch_size, seq_len = input_ids.size()
        labels_bsz, labels_seq_len = labels.size()
        # print(collated_batch)
        # print(seq_len)
        # first_row_input_ids = collated_batch['input_ids'][0]
        # first_row_labels = collated_batch['labels'][0]

        # # Print the full tensors without truncation
        # print("First row of input_ids:")
        # print(first_row_input_ids.tolist())  # Convert to list for complete printing

        # print("\nFirst row of labels:")
        # print(first_row_labels.tolist())  # Convert to list for complete printing
        
        # Pad to the fixed length if needed
        if seq_len < self.max_length:
            padding_len = self.max_length - seq_len           
            padding = torch.full((batch_size, padding_len), self.pad_token_id, 
                                dtype=input_ids.dtype, device=input_ids.device)
            input_ids = torch.cat([input_ids, padding], dim=1)
        elif seq_len > self.max_length:
            # Truncate if sequence is longer than max_length
            input_ids = input_ids[:, :self.max_length]

        if labels_seq_len < self.max_length:
            padding_len = self.max_length - labels_seq_len
            label_padding = torch.full((labels_bsz, padding_len), -100, 
                                      dtype=labels.dtype, device=labels.device)
            labels = torch.cat([labels, label_padding], dim=1)
        elif labels_seq_len > self.max_length:
            # Truncate if sequence is longer than max_length
            labels = labels[:, :self.max_length]

        # Return just the input_ids and labels as a tuple
        return {"input_ids": input_ids, "labels": labels}


def train_epoch(model, dataloader, optimizer, device, lr_scheduler, max_grad_norm=1.0):
    """Train the model for one epoch"""
    model.train()
    total_loss = 0
    start_time = time.time()
    
    # Progress bar
    pbar = tqdm(total=len(dataloader), desc="Training")
    
    # For gradient accumulation
    optimizer.zero_grad()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(f"Warning: Parameter {name} does not require gradients")
    print(f"Initial learning rate: {optimizer.param_groups[0]['lr']}")
    
    for step, batch in enumerate(dataloader):
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}

        # Regular forward and backward pass
        outputs = model(**batch)
        loss = outputs.loss
        # print(f"Loss requires grad: {loss.requires_grad}")
        loss.backward()
        if torch.isnan(outputs.logits).any():
            print("NaN in model outputs")
            break
        if torch.isnan(loss):
            print("NaN in loss calculation")
            # Print stats on the outputs
            print(f"Logits min/max/mean: {outputs.logits.min().item()}, {outputs.logits.max().item()}, {outputs.logits.mean().item()}")
            break

        # Add this to validate batch format
        # if step == 0 or step == 1:  # First batch
        #     print("Sample input:", batch["input_ids"][0][:20])
        #     print("Sample label:", batch["labels"][0][:20])
        #     print("Sample output:", torch.argmax(outputs.logits[0, :20], dim=-1))
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        
        # Update parameters
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        
        # Update total loss (get the original loss value)
        total_loss += loss.item()
        
        # Update progress bar
        pbar.update(1)
        pbar.set_postfix({"loss": loss.item()})
        
        # Log to wandb if enabled
        if wandb.run is not None and step % 10 == 0:
            wandb.log({"train/loss": loss.item(), 
                      "train/step": step + len(dataloader) * epoch})
    
    pbar.close()
    
    # Calculate average loss
    avg_loss = total_loss / len(dataloader)
    training_time = time.time() - start_time
    
    return avg_loss, training_time


def evaluate(model, dataloader, device):
    """Evaluate the model on the validation set"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluation"):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass with mixed precision if enabled
            outputs = model(**batch)
            
            total_loss += outputs.loss.item()
    
    # Calculate average loss
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss


@hydra.main(config_path="configs", config_name="config")
def main(config):
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    logger = logging.getLogger(__name__)
    
    # Initialize wandb if needed
    if config.get("wandb", False):
        wandb.init(project=config.get("wandb", "gpt2-midi"), 
                   name=config.get("wandb_run_name", "gpt2"),
                   config=OmegaConf.to_container(config, resolve=True))
    
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create model
    logger.info("Initializing model")
    model_config = GPT2Config(
        vocab_size=16000,
        n_positions=8192,
        n_embd=512,
        n_layer=8,
        n_head=8,
        n_inner=config.model.get("n_inner", 3 * config.model.n_embd),
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16
    )
    
    model = AutoModelForCausalLM.from_config(model_config)
    
    # Move model to device
    model.to(device)
    
    # Load dataset
    from dataset import MIDIDataset
    from datasets import load_dataset, load_from_disk
    from miditok import MMM
    
    dc = config.data
    
    # Initialize tokenizer
    logger.info("Initializing tokenizer")
    tokenizer = MMM(params=dc.tokenizer_path)
    config.model.vocab_size = tokenizer.vocab_size
    
    # Load or filter dataset
    logger.info("Loading dataset")
    if not os.path.exists(dc.prefiltered_dataset_path):
        logger.info("Filtering dataset")
        ds = load_dataset("parquet", data_files={
            "train": dc.otherwise_train_data_path,
            "validation": dc.otherwise_val_data_path,
        })
        
        def is_score_valid(music, min_bars, min_notes):
            # This function should be imported or defined based on your implementation
            # For now, we'll assume all scores are valid
            return True
        
        ds = ds.filter(
            lambda ex: is_score_valid(
                ex["music"], dc.min_num_bars_file_valid, dc.min_num_notes_file_valid
            )
        )
        
        # Save the filtered dataset
        ds.save_to_disk(dc.prefiltered_dataset_path)
        logger.info("Filtered dataset saved to disk")
    else:
        logger.info(f"Loading pre-filtered dataset from {dc.prefiltered_dataset_path}")
        ds = load_from_disk(dc.prefiltered_dataset_path)
    
    # Create train and validation datasets
    tokenizer = MMM(params=dc.tokenizer_path)  # MMM(TokenizerConfig(**deepcopy(TOKENIZER_PARAMS)))
    config.model.vocab_size = tokenizer.vocab_size
    dc.tracks_selection_random_ratio_range = (0.4, 1)
    dc.ratios_range_bar_infilling_duration = (0.1, 0.4)
    dc.acs_random_ratio_range = (0.05, 0.9)
    dc.tracks_idx_random_ratio_range = (0.1, 1)
    dc.bars_idx_random_ratio_range = (0.1, 0.7)
    dc.data_augmentation_offsets = (6, 2, 0) 
    logger.info("Creating train and validation datasets")
    train_data = MIDIDataset(
        ds["train"],
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
    
    # Create collator
    collator = DataCollatorNoneFilter(pad_token_id=tokenizer.pad_token_id, max_length=config.model.ctx_len)
    
    # Create DataLoader
    train_loader = DataLoader(
        train_data,
        batch_size=8,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        collate_fn=collator
    )
    
    val_loader = DataLoader(
        val_data,
        batch_size=8,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
        collate_fn=collator
    )
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=config.training.weight_decay,
        betas=(config.training.get("beta1", 0.9), config.training.get("beta2", 0.999)),
        eps=config.training.get("adam_epsilon", 1e-8)
    )
    
    # Create learning rate scheduler
    total_steps = len(train_loader)
    warmup_steps = config.training.get("warmup_steps", 1000)
    
    from transformers import get_scheduler
    lr_scheduler = get_scheduler(
        config.training.get("lr_scheduler_type", "cosine"),
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # Training loop
    logger.info("Starting training")
    best_val_loss = float('inf')
    
    # Create output directory if it doesn't exist
    output_dir = "/home/christian/MIDI-RWKV/src/outputs/gpt2"
    os.makedirs(output_dir, exist_ok=True)
    
    global epoch  # Make epoch global for wandb logging in train_epoch function
    for epoch in range(1):
        logger.info(f"Epoch {epoch+1}/{1}")
        
        # Train
        train_loss, train_time = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            lr_scheduler=lr_scheduler,
            max_grad_norm=config.training.get("max_grad_norm", 1.0)
        )
                
        logger.info(f"Train Loss: {train_loss:.4f}, Time: {train_time:.2f}s")
        
        # Evaluate if validation is enabled
        if config.training.get("do_eval", True):
            val_loss = evaluate(
                model=model,
                dataloader=val_loader,
                device=device,
            )
            
            logger.info(f"Validation Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                logger.info(f"New best validation loss: {best_val_loss:.4f}")
                
                # Save model
                model_path = os.path.join(output_dir, "best_model")
                model.save_pretrained(model_path)
                logger.info(f"Best model saved to {model_path}")
        
        # Save checkpoint at the end of each epoch
        checkpoint_path = os.path.join(output_dir, f"checkpoint-epoch-{epoch+1}")
        model.save_pretrained(checkpoint_path)
        
        # Log to wandb if enabled
        if wandb.run is not None:
            wandb.log({
                "train/epoch_loss": train_loss,
                "train/epoch": epoch + 1,
                "train/learning_rate": optimizer.param_groups[0]["lr"],
            })
            
            if config.training.get("do_eval", True):
                wandb.log({"eval/loss": val_loss, "eval/epoch": epoch + 1})
    
    # Save final model
    model_path = os.path.join(output_dir, "final_model")
    model.save_pretrained(model_path)
    logger.info(f"Final model saved to {model_path}")
    
    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()