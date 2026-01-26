#!/bin/bash
# Distributed training launcher for Crowd-BT with pre-prepared data
#
# Usage:
#   ./train_distributed_prepared.sh <num_gpus> [additional args...]
#
# Example:
#   ./train_distributed_prepared.sh 2 --data-dir ./training_data --image-dir ./data/images --epochs 50

# Check if number of GPUs is provided
if [ -z "$1" ]; then
    echo "Error: Number of GPUs not specified"
    echo "Usage: $0 <num_gpus> [additional args...]"
    echo "Example: $0 2 --data-dir ./training_data --image-dir ./data/images --epochs 50"
    exit 1
fi

NUM_GPUS=$1
shift  # Remove first argument so $@ contains only additional args

echo "=========================================="
echo "Starting Distributed Training (Prepared Data)"
echo "=========================================="
echo "Number of GPUs: $NUM_GPUS"
echo "Additional arguments: $@"
echo "=========================================="

# Use torchrun (recommended for PyTorch 1.10+)
if command -v torchrun &> /dev/null; then
    echo "Using torchrun..."
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --master_port=29500 \
        train_from_prepared_data.py \
        --distributed \
        --world-size $NUM_GPUS \
        "$@"
# Fallback to python -m torch.distributed.launch for older PyTorch versions
elif python -c "import torch.distributed.launch" 2>/dev/null; then
    echo "Using torch.distributed.launch..."
    python -m torch.distributed.launch \
        --nproc_per_node=$NUM_GPUS \
        --master_port=29500 \
        train_from_prepared_data.py \
        --distributed \
        --world-size $NUM_GPUS \
        "$@"
else
    echo "Error: Neither torchrun nor torch.distributed.launch is available"
    echo "Please install PyTorch with distributed support"
    exit 1
fi
