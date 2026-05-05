#!/bin/bash
# Launch script for Qwen Multi-Task Security Expert DDP Training

set -e

echo "=========================================="
echo "Qwen Multi-Task Security Expert"
echo "DDP Training Launcher"
echo "=========================================="

# Default values (auto-detect GPUs if not specified)
AVAILABLE_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo "0")
NUM_GPUS=${NUM_GPUS:-$AVAILABLE_GPUS}
CONFIG_FILE=${CONFIG_FILE:-"config/multitask_training_config.yaml"}
MASTER_PORT=${MASTER_PORT:-29500}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --master_port)
            MASTER_PORT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--num_gpus N] [--config path] [--master_port port]"
            exit 1
            ;;
    esac
done

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Validate GPU count
if [ "$NUM_GPUS" -eq 0 ]; then
    echo "Error: No GPUs detected. Please check nvidia-smi."
    exit 1
fi

if [ "$NUM_GPUS" -gt "$AVAILABLE_GPUS" ]; then
    echo "Warning: Requested $NUM_GPUS GPUs but only $AVAILABLE_GPUS available"
    echo "Using $AVAILABLE_GPUS GPUs instead"
    NUM_GPUS=$AVAILABLE_GPUS
fi

echo "Detected GPUs: $AVAILABLE_GPUS"

echo "Training Configuration:"
echo "  - GPUs: $NUM_GPUS"
echo "  - Config: $CONFIG_FILE"
echo "  - Master Port: $MASTER_PORT"
echo "=========================================="

# Create checkpoint directory
mkdir -p checkpoints

# Set environment variables
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # Reduce memory fragmentation

# Launch training
echo "Starting training with torchrun..."
echo ""

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    train_qwen_ddp.py \
    --config "$CONFIG_FILE"

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="
