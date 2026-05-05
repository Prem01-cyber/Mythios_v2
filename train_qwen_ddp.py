#!/usr/bin/env python3
"""
Distributed Data Parallel (DDP) Multi-Task Fine-Tuning for Qwen Security Expert

This script fine-tunes Qwen using PyTorch DDP across multiple GPUs on 4 security tasks:
1. [CLASSIFY] - Binary vulnerability classification
2. [CVE_LOOKUP] - CVE identification and analysis
3. [CODE_ANALYSIS] - Deep vulnerability analysis
4. [FIX] - Remediation guidance

Supports task-specific loss weighting and per-task metrics tracking.

Usage:
    # Single node, 4 GPUs
    torchrun --nproc_per_node=4 train_qwen_ddp.py --config config/multitask_training_config.yaml
    
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
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import logging
from tqdm import tqdm
import time
import wandb
from datetime import datetime


# Configure logging (simple format, rank added dynamically later)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


@dataclass
class TrainingConfig:
    """Training configuration for multi-task learning"""
    # Model
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    
    # Data
    train_data_path: str = "vuln_data/multitask_data/train.jsonl"
    val_data_path: str = "vuln_data/multitask_data/val.jsonl"
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
    
    # Multi-task settings
    task_weights: Dict[str, float] = field(default_factory=lambda: {
        "CLASSIFY": 0.20,
        "CVE_LOOKUP": 0.40,
        "CODE_ANALYSIS": 0.30,
        "FIX": 0.10
    })
    
    # DDP settings
    backend: str = "nccl"  # nccl for NVIDIA GPUs
    
    # Checkpointing
    output_dir: str = "checkpoints/qwen-security-expert-multitask"
    save_steps: int = 2000
    save_total_limit: int = 3
    
    # Logging
    logging_steps: int = 50
    eval_steps: int = 1000
    use_wandb: bool = True
    wandb_project: str = "qwen-multitask-security"
    
    # Mixed precision
    fp16: bool = True  # Use mixed precision training
    
    # Quantization settings (for memory-constrained GPUs)
    use_quantization: bool = False
    quantization_bits: int = 4  # 4-bit or 8-bit
    quantization_type: str = "nf4"  # "nf4" or "fp4"
    use_double_quant: bool = True  # Double quantization for extra memory savings
    
    # Memory optimizations
    gradient_checkpointing: bool = False  # Trade compute for memory
    
    # LoRA/PEFT settings (for efficient fine-tuning)
    use_lora: bool = False
    lora_r: int = 16  # LoRA rank
    lora_alpha: int = 32  # LoRA alpha scaling
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # Optimization
    optim: str = "adamw_torch"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    
    # Other
    seed: int = 42
    dataloader_num_workers: int = 4


class MultiTaskSecurityDataset(Dataset):
    """
    Dataset for multi-task security training
    Handles: CLASSIFY, CVE_LOOKUP, CODE_ANALYSIS, FIX tasks
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 2048,
        rank: int = 0
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        if rank == 0:
            logging.info(f"Loading dataset from {data_path}...")
        
        # Load data with progress tracking
        self.examples = []
        with open(data_path, 'r') as f:
            for i, line in enumerate(f):
                example = json.loads(line)
                self.examples.append(example)
                
                # Progress update every 100k examples
                if rank == 0 and (i + 1) % 100000 == 0:
                    logging.info(f"  Loaded {i+1:,} examples...")
        
        # Count tasks
        task_counts = {}
        for ex in self.examples:
            task = ex.get('task_type', 'UNKNOWN')
            task_counts[task] = task_counts.get(task, 0) + 1
        
        if rank == 0:
            logging.info(f"✓ Loaded {len(self.examples):,} examples from {data_path}")
            logging.info("Task distribution:")
            for task, count in sorted(task_counts.items()):
                pct = (count / len(self.examples)) * 100
                logging.info(f"  {task:20s}: {count:8,} ({pct:5.1f}%)")
    
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
        
        # Get task type for loss weighting
        task_type = example.get('task_type', 'UNKNOWN')
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'task_type': task_type
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


def load_model_and_tokenizer(config: TrainingConfig, device, rank: int = 0):
    """Load Qwen model and tokenizer with optional quantization"""
    if rank == 0:
        logging.info(f"Loading model: {config.model_name}")
        
        if config.use_quantization:
            logging.info(f"Using {config.quantization_bits}-bit quantization ({config.quantization_type})")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        padding_side='right'
    )
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare quantization config if enabled
    quantization_config = None
    if config.use_quantization:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=(config.quantization_bits == 4),
            load_in_8bit=(config.quantization_bits == 8),
            bnb_4bit_quant_type=config.quantization_type,  # "nf4" or "fp4"
            bnb_4bit_compute_dtype=torch.float16 if config.fp16 else torch.float32,
            bnb_4bit_use_double_quant=config.use_double_quant,  # Double quantization
        )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        quantization_config=quantization_config,
        torch_dtype=torch.float16 if config.fp16 else torch.float32,
        device_map=None if not config.use_quantization else {"": device}  # Quantized models need device_map
    )
    
    # Only move to device if not using quantization (quantization handles device placement)
    if not config.use_quantization:
        model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9
    
    if rank == 0:
        logging.info(f"Model loaded: {total_params:.2f}B total parameters, {trainable_params:.2f}B trainable")
        
        if config.use_quantization:
            logging.info(f"Memory footprint reduced by ~{100 * (1 - config.quantization_bits/16):.0f}% with {config.quantization_bits}-bit quantization")
    
    # Enable gradient checkpointing if requested
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if rank == 0:
            logging.info("✓ Gradient checkpointing enabled (trades compute for ~40% memory savings)")
    
    # Apply LoRA if enabled
    if config.use_lora:
        if rank == 0:
            logging.info(f"Applying LoRA (rank={config.lora_r}, alpha={config.lora_alpha})...")
        
        # Prepare model for k-bit training if quantized
        if config.use_quantization:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=config.gradient_checkpointing)
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # Apply LoRA
        model = get_peft_model(model, lora_config)
        
        if rank == 0:
            # Count trainable parameters after LoRA
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
            total_params = sum(p.numel() for p in model.parameters()) / 1e9
            trainable_pct = (trainable_params * 1e6) / (total_params * 1e9) * 100
            
            logging.info(f"✓ LoRA applied: {trainable_params:.1f}M trainable params ({trainable_pct:.2f}% of {total_params:.2f}B total)")
            model.print_trainable_parameters()
    
    return model, tokenizer


def create_dataloaders(
    config: TrainingConfig,
    tokenizer,
    rank: int,
    world_size: int
):
    """Create distributed dataloaders for multi-task training"""
    
    if rank == 0:
        logging.info("Creating datasets...")
    
    # Create datasets
    train_dataset = MultiTaskSecurityDataset(
        data_path=config.train_data_path,
        tokenizer=tokenizer,
        max_length=config.max_seq_length,
        rank=rank
    )
    
    val_dataset = MultiTaskSecurityDataset(
        data_path=config.val_data_path,
        tokenizer=tokenizer,
        max_length=config.max_seq_length,
        rank=rank
    )
    
    if rank == 0:
        logging.info("Creating distributed samplers...")
    
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
    
    if rank == 0:
        logging.info("Creating dataloaders...")
    
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
    
    if rank == 0:
        logging.info(f"✓ Dataloaders created (train: {len(train_loader):,} batches, val: {len(val_loader):,} batches)")
    
    return train_loader, val_loader, train_sampler


def create_optimizer_and_scheduler(
    model,
    config: TrainingConfig,
    num_training_steps: int,
    rank: int = 0
):
    """Create optimizer and learning rate scheduler"""
    
    if rank == 0:
        logging.info("Creating optimizer and scheduler...")
    
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
    
    if rank == 0:
        logging.info(f"✓ Optimizer: AdamW (LR={config.learning_rate:.2e})")
        logging.info(f"✓ Scheduler: Cosine with {config.warmup_steps} warmup steps")
    
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


def calculate_task_weighted_loss(
    loss: torch.Tensor,
    task_types: List[str],
    task_weights: Dict[str, float]
) -> torch.Tensor:
    """
    Calculate task-weighted loss for multi-task learning
    
    Args:
        loss: Base loss from model
        task_types: List of task types in batch
        task_weights: Dict mapping task to weight
    
    Returns:
        Weighted loss
    """
    if not task_weights or len(task_types) == 0:
        return loss
    
    # Calculate average weight for this batch
    batch_weights = [task_weights.get(task, 1.0) for task in task_types]
    avg_weight = sum(batch_weights) / len(batch_weights)
    
    return loss * avg_weight


def evaluate(model, val_loader, device, rank, config: TrainingConfig = None):
    """Run evaluation with per-task losses and metrics"""
    model.eval()
    total_loss = 0
    total_steps = 0
    
    # Per-task metrics
    task_losses = {}
    task_counts = {}
    
    # Task-specific accuracy tracking
    classify_correct = 0
    classify_total = 0
    
    with torch.no_grad():
        for batch in val_loader:
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            task_types = batch.get('task_type', [])
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # Track per-task loss
            if isinstance(task_types, list):
                for task in task_types:
                    if task not in task_losses:
                        task_losses[task] = 0
                        task_counts[task] = 0
                    task_losses[task] += loss.item()
                    task_counts[task] += 1
                    
                    # CLASSIFY task: compute token-level accuracy
                    # Note: This measures next-token prediction accuracy across the whole sequence
                    # For true classification accuracy (0/1 prediction), use evaluate_multitask.py
                    if task == 'CLASSIFY':
                        logits = outputs.logits  # [batch, seq_len, vocab_size]
                        predictions = torch.argmax(logits, dim=-1)  # [batch, seq_len]
                        
                        # Compare predictions with labels for non-padded positions
                        mask = labels != -100
                        if mask.any():
                            correct = (predictions[mask] == labels[mask]).sum().item()
                            total = mask.sum().item()
                            classify_correct += correct
                            classify_total += total
            
            total_loss += loss.item()
            total_steps += 1
    
    avg_loss = total_loss / total_steps if total_steps > 0 else 0
    
    # Calculate per-task average losses
    task_avg_losses = {}
    for task in task_losses:
        if task_counts[task] > 0:
            task_avg_losses[task] = task_losses[task] / task_counts[task]
    
    # Add CLASSIFY token accuracy if available
    # Note: This is next-token prediction accuracy, not binary classification accuracy
    if classify_total > 0:
        task_avg_losses['CLASSIFY_TOKEN_ACC'] = classify_correct / classify_total
    
    # Average across all GPUs
    if dist.is_initialized():
        avg_loss_tensor = torch.tensor(avg_loss, device=device)
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.AVG)
        avg_loss = avg_loss_tensor.item()
        
        # Sync classify token accuracy metrics
        if classify_total > 0:
            classify_tensors = torch.tensor([classify_correct, classify_total], dtype=torch.float32, device=device)
            dist.all_reduce(classify_tensors, op=dist.ReduceOp.SUM)
            task_avg_losses['CLASSIFY_TOKEN_ACC'] = (classify_tensors[0] / classify_tensors[1]).item()
    
    return avg_loss, task_avg_losses


def train(config: TrainingConfig):
    """Main training function"""
    
    # Setup DDP
    rank, local_rank, world_size, device = setup_ddp()
    
    if rank == 0:
        logging.info("="*60)
        logging.info("Qwen Multi-Task Security Expert - DDP Fine-Tuning")
        logging.info("="*60)
        logging.info(f"World Size: {world_size}")
        logging.info(f"Per-device batch size: {config.per_device_train_batch_size}")
        logging.info(f"Gradient accumulation: {config.gradient_accumulation_steps}")
        logging.info(f"Effective batch size: {config.per_device_train_batch_size * world_size * config.gradient_accumulation_steps}")
        logging.info(f"Learning rate: {config.learning_rate}")
        logging.info(f"Epochs: {config.num_epochs}")
        
        if config.task_weights:
            logging.info("\nTask Weights:")
            for task, weight in sorted(config.task_weights.items()):
                logging.info(f"  {task:20s}: {weight:.2f}")
        
        if config.use_lora:
            logging.info(f"\nLoRA Settings:")
            logging.info(f"  Rank (r): {config.lora_r}")
            logging.info(f"  Alpha: {config.lora_alpha}")
            logging.info(f"  Dropout: {config.lora_dropout}")
            logging.info(f"  Target modules: {len(config.lora_target_modules)} layers")
        
        logging.info("="*60)
        
        # Initialize wandb
        if config.use_wandb:
            wandb.init(
                project=config.wandb_project,
                config=config.__dict__,
                name=f"qwen-cve-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                mode="disabled"  # Disable W&B to prevent interactive prompts
            )
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config, device, rank)
    
    if rank == 0:
        logging.info("✓ Model loaded successfully")
    
    # Clear GPU cache before DDP to free memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Wrap with DDP
    if world_size > 1:
        if rank == 0:
            logging.info("Wrapping model with DDP...")
        
        # For LoRA + quantization, use special DDP settings to avoid OOM
        if config.use_lora and config.use_quantization:
            if rank == 0:
                logging.info("Using memory-optimized DDP settings for LoRA + quantization")
            
            # Only synchronize LoRA parameters, not the frozen base model
            model = DDP(
                model, 
                device_ids=[local_rank], 
                output_device=local_rank,
                find_unused_parameters=True,  # Only sync trainable params
                broadcast_buffers=False,  # Don't sync buffers
                bucket_cap_mb=1  # Use very small buckets (default 25MB)
            )
        else:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        
        # Enable static graph for DDP + gradient checkpointing compatibility
        if config.gradient_checkpointing:
            model._set_static_graph()
            if rank == 0:
                logging.info("✓ Static graph enabled for DDP + gradient checkpointing")
        
        if rank == 0:
            logging.info("✓ DDP initialized")
    
    # Create dataloaders
    train_loader, val_loader, train_sampler = create_dataloaders(
        config, tokenizer, rank, world_size
    )
    
    # Calculate total steps
    steps_per_epoch = len(train_loader) // config.gradient_accumulation_steps
    total_steps = steps_per_epoch * config.num_epochs
    
    if rank == 0:
        logging.info(f"\nTraining schedule:")
        logging.info(f"  Steps per epoch: {steps_per_epoch:,}")
        logging.info(f"  Total training steps: {total_steps:,}")
        logging.info(f"  Estimated time: {total_steps * config.gradient_accumulation_steps / (200 * world_size):.1f} hours")
    
    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(
        model, config, total_steps, rank
    )
    
    if rank == 0:
        logging.info("\n" + "="*60)
        logging.info("Starting training...")
        logging.info("="*60)
    
    # Training loop
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(config.num_epochs):
        if rank == 0:
            logging.info(f"\n{'='*60}")
            logging.info(f"EPOCH {epoch+1}/{config.num_epochs}")
            logging.info(f"{'='*60}")
        
        # Set epoch for sampler (important for shuffling)
        train_sampler.set_epoch(epoch)
        
        model.train()
        epoch_loss = 0
        optimizer.zero_grad()
        
        if rank == 0:
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}", ncols=100)
        else:
            progress_bar = train_loader
        
        # Per-task loss tracking
        task_losses = {}
        task_counts = {}
        
        for step, batch in enumerate(progress_bar):
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            task_types = batch.get('task_type', [])
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            base_loss = outputs.loss
            
            # Apply task weighting if task_weights are provided
            if config.task_weights and task_types:
                weighted_loss = calculate_task_weighted_loss(
                    base_loss,
                    task_types,
                    config.task_weights
                )
            else:
                weighted_loss = base_loss
            
            loss = weighted_loss / config.gradient_accumulation_steps
            loss.backward()
            
            epoch_loss += loss.item()
            
            # Track per-task loss
            if task_types:
                for task in task_types:
                    if task not in task_losses:
                        task_losses[task] = 0
                        task_counts[task] = 0
                    task_losses[task] += base_loss.item()
                    task_counts[task] += 1
            
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
                    
                    # Calculate per-task losses
                    task_avg_losses = {}
                    for task in task_losses:
                        if task_counts[task] > 0:
                            task_avg_losses[task] = task_losses[task] / task_counts[task]
                    
                    # Calculate progress percentage
                    progress_pct = (global_step / total_steps) * 100
                    
                    log_msg = (
                        f"[{progress_pct:5.1f}%] Step {global_step:,}/{total_steps:,} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"LR: {current_lr:.2e}"
                    )
                    
                    # Add per-task losses to log
                    if task_avg_losses:
                        task_loss_str = " | ".join([f"{task[:8]}: {loss:.3f}" 
                                                     for task, loss in sorted(task_avg_losses.items())])
                        log_msg += f" | {task_loss_str}"
                    
                    logging.info(log_msg)
                    
                    if config.use_wandb:
                        wandb_log = {
                            'train/loss': avg_loss,
                            'train/learning_rate': current_lr,
                            'train/epoch': epoch,
                            'train/global_step': global_step
                        }
                        # Add per-task losses
                        for task, loss in task_avg_losses.items():
                            wandb_log[f'train/loss_{task}'] = loss
                        
                        wandb.log(wandb_log)
                    
                    epoch_loss = 0
                    task_losses.clear()
                    task_counts.clear()
                
                # Evaluation
                if global_step % config.eval_steps == 0:
                    if rank == 0:
                        logging.info("Running evaluation...")
                    
                    val_loss, val_task_losses = evaluate(model, val_loader, device, rank, config)
                    
                    if rank == 0:
                        log_msg = f"Validation Loss: {val_loss:.4f}"
                        
                        # Add per-task validation losses
                        if val_task_losses:
                            task_loss_str = " | ".join([f"{task}: {loss:.4f}" 
                                                         for task, loss in sorted(val_task_losses.items())])
                            log_msg += f" | Task Losses: {task_loss_str}"
                        
                        logging.info(log_msg)
                        
                        if config.use_wandb:
                            wandb_log = {
                                'val/loss': val_loss,
                                'val/step': global_step
                            }
                            # Add per-task validation losses
                            for task, loss in val_task_losses.items():
                                wandb_log[f'val/loss_{task}'] = loss
                            
                            wandb.log(wandb_log)
                        
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
    
    final_val_loss, final_task_losses = evaluate(model, val_loader, device, rank, config)
    
    if rank == 0:
        logging.info("="*60)
        logging.info("Training Complete!")
        logging.info(f"Final Validation Loss: {final_val_loss:.4f}")
        logging.info(f"Best Validation Loss: {best_val_loss:.4f}")
        
        if final_task_losses:
            logging.info("\nFinal Per-Task Losses:")
            for task, loss in sorted(final_task_losses.items()):
                logging.info(f"  {task:20s}: {loss:.4f}")
        
        logging.info("="*60)
        
        if config.use_wandb:
            wandb_log = {
                'final/val_loss': final_val_loss,
                'final/best_val_loss': best_val_loss
            }
            for task, loss in final_task_losses.items():
                wandb_log[f'final/loss_{task}'] = loss
            
            wandb.log(wandb_log)
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
