#!/bin/bash
# Launch script for DeepSpeed ZeRO-3 Training (Model Parallelism)
# Model is SPLIT across GPUs with FULL PRECISION (no quantization)

set -e

echo "=========================================="
echo "Qwen Multi-Task Security Expert"
echo "DeepSpeed ZeRO-3 Training (Model Parallel)"
echo "=========================================="

# Default values
AVAILABLE_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo "0")
NUM_GPUS=${NUM_GPUS:-$AVAILABLE_GPUS}
CONFIG_FILE=${CONFIG_FILE:-"config/multitask_training_config_zero3.yaml"}

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
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--num_gpus N] [--config path]"
            exit 1
            ;;
    esac
done

# Validate
if [ "$NUM_GPUS" -eq 0 ]; then
    echo "Error: No GPUs detected"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "Training Configuration:"
echo "  - GPUs: $NUM_GPUS"
echo "  - Config: $CONFIG_FILE"
echo "  - Mode: Model Parallelism (ZeRO-3)"
echo "  - Precision: Full fp16 (NO quantization)"
echo "=========================================="

# Create checkpoint directory
mkdir -p checkpoints

# Set environment variables
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Install DeepSpeed if not already installed
if ! python -c "import deepspeed" 2>/dev/null; then
    echo ""
    echo "Installing DeepSpeed..."
    pip install deepspeed>=0.12.0
    echo "✓ DeepSpeed installed"
    echo ""
fi

# Launch training with DeepSpeed
echo "Starting training with DeepSpeed ZeRO-3..."
echo ""

deepspeed \
    --num_gpus=$NUM_GPUS \
    train_qwen_zero3.py \
    --config "$CONFIG_FILE"

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="
