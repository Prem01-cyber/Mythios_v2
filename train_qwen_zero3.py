#!/usr/bin/env python3
"""
DeepSpeed ZeRO-3 Multi-Task Fine-Tuning for Qwen Security Expert

This script uses MODEL PARALLELISM to split the model across GPUs
for FULL PRECISION training (fp16) instead of quantization.

Key differences from DDP approach:
- Model split across GPUs (not replicated)
- Full fp16 precision (not 4-bit quantization)
- Better model quality
- Uses DeepSpeed ZeRO-3 for efficient model sharding

Usage:
    deepspeed --num_gpus=4 train_qwen_zero3.py --config config/multitask_training_config_zero3.yaml
"""

import os
import json
import torch
import deepspeed
from deepspeed import comm as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import logging
from tqdm import tqdm
import time
from datetime import datetime

# Speed optimizations
torch.set_float32_matmul_precision('medium')
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


@dataclass
class TrainingConfig:
    """Training configuration for DeepSpeed ZeRO-3"""
    # Model
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    
    # Data
    train_data_path: str = "vuln_data/multitask_data/train.jsonl"
    val_data_path: str = "vuln_data/multitask_data/val.jsonl"
    max_seq_length: int = 512
    
    # Training hyperparameters
    num_epochs: int = 3
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    
    # Multi-task weights
    task_weights: Optional[Dict[str, float]] = None
    
    # DeepSpeed
    use_deepspeed: bool = True
    deepspeed_config: str = "config/deepspeed_zero3_config.json"
    local_rank: int = -1
    
    # Checkpointing
    output_dir: str = "checkpoints/qwen-security-expert-multitask-zero3"
    save_steps: int = 5000
    save_total_limit: int = 3
    
    # Logging
    logging_steps: int = 100
    eval_steps: int = 5000
    use_wandb: bool = False
    wandb_project: str = "qwen-multitask-security-zero3"
    
    # Optimization (full precision, no quantization)
    fp16: bool = True
    optim: str = "adamw_torch"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    
    # NO quantization for ZeRO-3
    use_quantization: bool = False
    
    # Memory optimizations
    gradient_checkpointing: bool = True
    use_flash_attention_2: bool = True
    
    # LoRA settings (optional, can disable for full fine-tuning)
    use_lora: bool = True
    lora_r: int = 32  # Higher than quantized version
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # DataLoader
    seed: int = 42
    dataloader_num_workers: int = 12
    prefetch_factor: int = 4
    persistent_workers: bool = True


# Reuse dataset and collator from DDP script
class MultiTaskSecurityDataset(Dataset):
    """Dataset for multi-task security training"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 512, rank: int = 0):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        if rank == 0:
            logging.info(f"Loading dataset from {data_path}...")
        
        self.examples = []
        with open(data_path, 'r') as f:
            for i, line in enumerate(f):
                example = json.loads(line)
                self.examples.append(example)
                
                if rank == 0 and (i + 1) % 100000 == 0:
                    logging.info(f"  Loaded {i+1:,} examples...")
        
        if rank == 0:
            logging.info(f"✓ Loaded {len(self.examples):,} examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Format conversation
        formatted_text = ""
        for message in example['messages']:
            role = message['role']
            content = message['content']
            formatted_text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        
        # Tokenize without padding (dynamic padding in DataCollator)
        encoding = self.tokenizer(
            formatted_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        labels = input_ids.clone()
        task_type = example.get('task_type', 'UNKNOWN')
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'task_type': task_type
        }


class DataCollatorWithDynamicPadding:
    """Dynamic padding collator"""
    
    def __init__(self, tokenizer, pad_to_multiple_of=8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
    
    def __call__(self, batch):
        task_types = [item['task_type'] for item in batch]
        max_len = max(len(item['input_ids']) for item in batch)
        
        if self.pad_to_multiple_of:
            max_len = ((max_len + self.pad_to_multiple_of - 1) 
                      // self.pad_to_multiple_of * self.pad_to_multiple_of)
        
        input_ids = []
        attention_mask = []
        labels = []
        
        for item in batch:
            seq_len = len(item['input_ids'])
            pad_len = max_len - seq_len
            
            input_ids.append(
                torch.cat([
                    item['input_ids'],
                    torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=torch.long)
                ])
            )
            attention_mask.append(
                torch.cat([
                    item['attention_mask'],
                    torch.zeros(pad_len, dtype=torch.long)
                ])
            )
            labels.append(
                torch.cat([
                    item['labels'],
                    torch.full((pad_len,), -100, dtype=torch.long)
                ])
            )
        
        return {
            'input_ids': torch.stack(input_ids),
            'attention_mask': torch.stack(attention_mask),
            'labels': torch.stack(labels),
            'task_type': task_types
        }


def load_model_and_tokenizer(config: TrainingConfig, rank: int = 0):
    """Load Qwen model with full precision (no quantization)"""
    if rank == 0:
        logging.info(f"Loading model: {config.model_name}")
        logging.info("Using FULL PRECISION (fp16, no quantization)")
        logging.info("Model will be SPLIT across GPUs with DeepSpeed ZeRO-3")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        padding_side='right'
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with full precision
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.float16 if config.fp16 else torch.float32,
    }
    
    # Enable Flash Attention 2
    if config.use_flash_attention_2:
        try:
            model_kwargs["attn_implementation"] = "flash_attention_2"
            if rank == 0:
                logging.info("✓ Flash Attention 2 enabled")
        except Exception as e:
            if rank == 0:
                logging.warning(f"Flash Attention 2 not available: {e}")
    
    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    
    if rank == 0:
        logging.info(f"✓ Model loaded: {total_params:.2f}B parameters")
        logging.info(f"✓ Each GPU will hold ~{total_params / dist.get_world_size():.2f}B parameters")
    
    # Enable gradient checkpointing
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if rank == 0:
            logging.info("✓ Gradient checkpointing enabled")
    
    # Apply LoRA if enabled
    if config.use_lora:
        if rank == 0:
            logging.info(f"Applying LoRA (rank={config.lora_r}, alpha={config.lora_alpha})...")
        
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        model = get_peft_model(model, lora_config)
        
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        if rank == 0:
            logging.info(f"✓ LoRA applied: {trainable_params:.2f}M trainable parameters")
    
    return model, tokenizer


def train(config: TrainingConfig):
    """Main training function with DeepSpeed ZeRO-3"""
    
    # Initialize DeepSpeed
    deepspeed.init_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if rank == 0:
        logging.info("="*60)
        logging.info("Qwen Multi-Task Security Expert - DeepSpeed ZeRO-3")
        logging.info("="*60)
        logging.info(f"World Size: {world_size} GPUs")
        logging.info(f"Model Parallelism: Model SPLIT across GPUs")
        logging.info(f"Precision: Full fp16 (no quantization)")
        logging.info(f"Per-device batch size: {config.per_device_train_batch_size}")
        logging.info(f"Gradient accumulation: {config.gradient_accumulation_steps}")
        logging.info(f"Effective batch size: {config.per_device_train_batch_size * world_size * config.gradient_accumulation_steps}")
        logging.info("="*60)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config, rank)
    
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
    
    # Create dataloaders
    collator = DataCollatorWithDynamicPadding(tokenizer)
    
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=config.seed
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.per_device_train_batch_size,
        sampler=train_sampler,
        num_workers=config.dataloader_num_workers,
        pin_memory=True,
        collate_fn=collator,
        prefetch_factor=config.prefetch_factor,
        persistent_workers=config.persistent_workers
    )
    
    # Initialize DeepSpeed engine
    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        config=config.deepspeed_config,
        model_parameters=model.parameters()
    )
    
    if rank == 0:
        logging.info(f"✓ DeepSpeed engine initialized")
        logging.info(f"✓ Training ready!")
    
    # Training loop
    global_step = 0
    
    for epoch in range(config.num_epochs):
        if rank == 0:
            logging.info(f"\n{'='*60}")
            logging.info(f"EPOCH {epoch+1}/{config.num_epochs}")
            logging.info(f"{'='*60}")
        
        train_sampler.set_epoch(epoch)
        model_engine.train()
        
        if rank == 0:
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        else:
            progress_bar = train_loader
        
        for step, batch in enumerate(progress_bar):
            # Move to device
            input_ids = batch['input_ids'].cuda(non_blocking=True)
            attention_mask = batch['attention_mask'].cuda(non_blocking=True)
            labels = batch['labels'].cuda(non_blocking=True)
            
            # Forward pass
            outputs = model_engine(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # Backward pass
            model_engine.backward(loss)
            model_engine.step()
            
            global_step += 1
            
            # Logging
            if global_step % config.logging_steps == 0 and rank == 0:
                logging.info(f"Step {global_step:,} | Loss: {loss.item():.4f}")
            
            # Checkpointing
            if global_step % config.save_steps == 0:
                if rank == 0:
                    checkpoint_dir = Path(config.output_dir) / f"checkpoint-step{global_step}"
                    model_engine.save_checkpoint(checkpoint_dir)
                    logging.info(f"✓ Checkpoint saved: {checkpoint_dir}")
        
        if rank == 0:
            logging.info(f"Epoch {epoch+1} completed")
    
    if rank == 0:
        logging.info("\n" + "="*60)
        logging.info("Training Complete!")
        logging.info("="*60)


def main():
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--local_rank', type=int, default=-1)
    args, unknown = parser.parse_known_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    config_dict['local_rank'] = args.local_rank
    config = TrainingConfig(**config_dict)
    
    train(config)


if __name__ == '__main__':
    main()
