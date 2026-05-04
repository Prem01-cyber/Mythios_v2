#!/usr/bin/env python3
"""
Distributed Data Parallel (DDP) Fine-Tuning for Qwen on CVE Data

This script fine-tunes Qwen using PyTorch DDP across multiple GPUs to create
a vulnerability-aware security expert model.

Usage:
    # Single node, 4 GPUs
    torchrun --nproc_per_node=4 train_qwen_ddp.py --config config.yaml
    
    # Multi-node (if needed)
    torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 --master_addr=<addr> train_qwen_ddp.py
"""

import os
import json
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
    TrainingArguments
)
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import logging
from tqdm import tqdm
import time
import wandb
from datetime import datetime


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [Rank %(rank)s] - %(levelname)s - %(message)s'
)


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Model
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    
    # Data
    train_data_path: str = "vuln_data/instruction_data/train.jsonl"
    val_data_path: str = "vuln_data/instruction_data/val.jsonl"
    max_seq_length: int = 2048
    
    # Training hyperparameters
    num_epochs: int = 3
    per_device_train_batch_size: int = 2  # Per GPU
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    
    # DDP settings
    backend: str = "nccl"  # nccl for NVIDIA GPUs
    
    # Checkpointing
    output_dir: str = "checkpoints/qwen-cve-expert"
    save_steps: int = 2000
    save_total_limit: int = 3
    
    # Logging
    logging_steps: int = 50
    eval_steps: int = 1000
    use_wandb: bool = True
    wandb_project: str = "qwen-cve-finetuning"
    
    # Mixed precision
    fp16: bool = True  # Use mixed precision training
    
    # Optimization
    optim: str = "adamw_torch"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    
    # Other
    seed: int = 42
    dataloader_num_workers: int = 4


class CVEInstructionDataset(Dataset):
    """
    Dataset for CVE instruction-response pairs
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 2048
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data
        self.examples = []
        with open(data_path, 'r') as f:
            for line in f:
                self.examples.append(json.loads(line))
        
        logging.info(f"Loaded {len(self.examples)} examples from {data_path}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Format conversation for Qwen
        # Qwen format: <|im_start|>role\ncontent<|im_end|>
        formatted_text = ""
        for message in example['messages']:
            role = message['role']
            content = message['content']
            formatted_text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        
        # Tokenize
        encoding = self.tokenizer(
            formatted_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # For language modeling, labels = input_ids
        # but mask padding tokens (-100 tells PyTorch to ignore them in loss)
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


def setup_ddp():
    """Initialize DDP environment"""
    # Get rank from environment
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    # Initialize process group
    if world_size > 1:
        dist.init_process_group(backend='nccl')
    
    # Set device
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    
    return rank, local_rank, world_size, device


def cleanup_ddp():
    """Cleanup DDP"""
    if dist.is_initialized():
        dist.destroy_process_group()


def load_model_and_tokenizer(config: TrainingConfig, device):
    """Load Qwen model and tokenizer"""
    logging.info(f"Loading model: {config.model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        padding_side='right'
    )
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if config.fp16 else torch.float32,
        device_map=None  # We'll handle device placement with DDP
    )
    
    model = model.to(device)
    
    logging.info(f"Model loaded: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")
    
    return model, tokenizer


def create_dataloaders(
    config: TrainingConfig,
    tokenizer,
    rank: int,
    world_size: int
):
    """Create distributed dataloaders"""
    
    # Create datasets
    train_dataset = CVEInstructionDataset(
        data_path=config.train_data_path,
        tokenizer=tokenizer,
        max_length=config.max_seq_length
    )
    
    val_dataset = CVEInstructionDataset(
        data_path=config.val_data_path,
        tokenizer=tokenizer,
        max_length=config.max_seq_length
    )
    
    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=config.seed
    )
    
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.per_device_train_batch_size,
        sampler=train_sampler,
        num_workers=config.dataloader_num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.per_device_eval_batch_size,
        sampler=val_sampler,
        num_workers=config.dataloader_num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, train_sampler


def create_optimizer_and_scheduler(
    model,
    config: TrainingConfig,
    num_training_steps: int
):
    """Create optimizer and learning rate scheduler"""
    
    # Separate parameters for weight decay
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': config.weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_epsilon
    )
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=num_training_steps
    )
    
    return optimizer, scheduler


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    step,
    config: TrainingConfig,
    rank: int,
    best_val_loss: float = None
):
    """Save model checkpoint (only on rank 0)"""
    if rank != 0:
        return
    
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_dir = output_dir / f"checkpoint-epoch{epoch}-step{step}"
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Save model state (unwrap DDP)
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(checkpoint_dir)
    
    # Save training state
    training_state = {
        'epoch': epoch,
        'global_step': step,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss,
        'config': config.__dict__
    }
    torch.save(training_state, checkpoint_dir / 'training_state.pt')
    
    logging.info(f"Checkpoint saved to {checkpoint_dir}")
    
    # Manage checkpoint limit
    checkpoints = sorted(output_dir.glob('checkpoint-*'))
    if len(checkpoints) > config.save_total_limit:
        for old_checkpoint in checkpoints[:-config.save_total_limit]:
            logging.info(f"Removing old checkpoint: {old_checkpoint}")
            import shutil
            shutil.rmtree(old_checkpoint)


def evaluate(model, val_loader, device, rank):
    """Run evaluation"""
    model.eval()
    total_loss = 0
    total_steps = 0
    
    with torch.no_grad():
        for batch in val_loader:
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            total_steps += 1
    
    avg_loss = total_loss / total_steps if total_steps > 0 else 0
    
    # Average across all GPUs
    if dist.is_initialized():
        avg_loss_tensor = torch.tensor(avg_loss, device=device)
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.AVG)
        avg_loss = avg_loss_tensor.item()
    
    return avg_loss


def train(config: TrainingConfig):
    """Main training function"""
    
    # Setup DDP
    rank, local_rank, world_size, device = setup_ddp()
    
    # Update logging format with rank
    logging.LoggerAdapter(logging.getLogger(), {'rank': rank})
    
    if rank == 0:
        logging.info("="*60)
        logging.info("Qwen CVE Expert - DDP Fine-Tuning")
        logging.info("="*60)
        logging.info(f"World Size: {world_size}")
        logging.info(f"Per-device batch size: {config.per_device_train_batch_size}")
        logging.info(f"Gradient accumulation: {config.gradient_accumulation_steps}")
        logging.info(f"Effective batch size: {config.per_device_train_batch_size * world_size * config.gradient_accumulation_steps}")
        logging.info(f"Learning rate: {config.learning_rate}")
        logging.info(f"Epochs: {config.num_epochs}")
        logging.info("="*60)
        
        # Initialize wandb
        if config.use_wandb:
            wandb.init(
                project=config.wandb_project,
                config=config.__dict__,
                name=f"qwen-cve-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config, device)
    
    # Wrap with DDP
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    # Create dataloaders
    train_loader, val_loader, train_sampler = create_dataloaders(
        config, tokenizer, rank, world_size
    )
    
    # Calculate total steps
    steps_per_epoch = len(train_loader) // config.gradient_accumulation_steps
    total_steps = steps_per_epoch * config.num_epochs
    
    if rank == 0:
        logging.info(f"Training steps per epoch: {steps_per_epoch}")
        logging.info(f"Total training steps: {total_steps}")
    
    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(
        model, config, total_steps
    )
    
    # Training loop
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(config.num_epochs):
        # Set epoch for sampler (important for shuffling)
        train_sampler.set_epoch(epoch)
        
        model.train()
        epoch_loss = 0
        optimizer.zero_grad()
        
        if rank == 0:
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        else:
            progress_bar = train_loader
        
        for step, batch in enumerate(progress_bar):
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss / config.gradient_accumulation_steps
            loss.backward()
            
            epoch_loss += loss.item()
            
            # Gradient accumulation
            if (step + 1) % config.gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                
                # Optimizer step
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                global_step += 1
                
                # Logging
                if global_step % config.logging_steps == 0 and rank == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    avg_loss = epoch_loss / config.logging_steps
                    
                    logging.info(
                        f"Step {global_step}/{total_steps} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"LR: {current_lr:.2e}"
                    )
                    
                    if config.use_wandb:
                        wandb.log({
                            'train/loss': avg_loss,
                            'train/learning_rate': current_lr,
                            'train/epoch': epoch,
                            'train/global_step': global_step
                        })
                    
                    epoch_loss = 0
                
                # Evaluation
                if global_step % config.eval_steps == 0:
                    if rank == 0:
                        logging.info("Running evaluation...")
                    
                    val_loss = evaluate(model, val_loader, device, rank)
                    
                    if rank == 0:
                        logging.info(f"Validation Loss: {val_loss:.4f}")
                        
                        if config.use_wandb:
                            wandb.log({
                                'val/loss': val_loss,
                                'val/step': global_step
                            })
                        
                        # Save best model
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            logging.info(f"New best validation loss: {best_val_loss:.4f}")
                            save_checkpoint(
                                model, optimizer, scheduler,
                                epoch, global_step, config, rank, best_val_loss
                            )
                    
                    model.train()
                
                # Checkpointing
                if global_step % config.save_steps == 0:
                    save_checkpoint(
                        model, optimizer, scheduler,
                        epoch, global_step, config, rank, best_val_loss
                    )
        
        # End of epoch
        if rank == 0:
            logging.info(f"Epoch {epoch+1} completed")
            
        # Save epoch checkpoint
        save_checkpoint(
            model, optimizer, scheduler,
            epoch + 1, global_step, config, rank, best_val_loss
        )
    
    # Final evaluation
    if rank == 0:
        logging.info("Running final evaluation...")
    
    final_val_loss = evaluate(model, val_loader, device, rank)
    
    if rank == 0:
        logging.info("="*60)
        logging.info("Training Complete!")
        logging.info(f"Final Validation Loss: {final_val_loss:.4f}")
        logging.info(f"Best Validation Loss: {best_val_loss:.4f}")
        logging.info("="*60)
        
        if config.use_wandb:
            wandb.log({
                'final/val_loss': final_val_loss,
                'final/best_val_loss': best_val_loss
            })
            wandb.finish()
    
    # Cleanup
    cleanup_ddp()


def main():
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description='Fine-tune Qwen with DDP')
    parser.add_argument('--config', type=str, help='Path to config YAML file')
    parser.add_argument('--train_data', type=str, help='Override training data path')
    parser.add_argument('--val_data', type=str, help='Override validation data path')
    parser.add_argument('--output_dir', type=str, help='Override output directory')
    parser.add_argument('--epochs', type=int, help='Override number of epochs')
    parser.add_argument('--batch_size', type=int, help='Override batch size')
    parser.add_argument('--lr', type=float, help='Override learning rate')
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        with open(args.config) as f:
            config_dict = yaml.safe_load(f)
        config = TrainingConfig(**config_dict)
    else:
        config = TrainingConfig()
    
    # Override with command line args
    if args.train_data:
        config.train_data_path = args.train_data
    if args.val_data:
        config.val_data_path = args.val_data
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.epochs:
        config.num_epochs = args.epochs
    if args.batch_size:
        config.per_device_train_batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr
    
    # Run training
    train(config)


if __name__ == '__main__':
    main()
