# Distributed Training Guide

This guide explains how to use distributed training with the Crowd-BT Quality Scorer to train across multiple GPUs.

## Overview

Distributed training enables you to:
- Train on multiple GPUs simultaneously
- Reduce training time significantly
- Handle larger effective batch sizes
- Scale to multiple nodes (with additional configuration)

The implementation uses PyTorch's DistributedDataParallel (DDP) for efficient multi-GPU training.

## Quick Start

### Single Node, Multiple GPUs

The easiest way to launch distributed training is using the provided script:

```bash
# Train on 4 GPUs
./train_distributed.sh 4 --epochs 20 --batch-size 16

# Train on 2 GPUs with custom settings
./train_distributed.sh 2 --data-source s3 --backbone efficientnet_b5 --lr 5e-5
```

### Manual Launch

You can also manually launch distributed training using `torchrun`:

```bash
torchrun --nproc_per_node=4 train.py \
    --distributed \
    --world-size 4 \
    --epochs 20 \
    --batch-size 16
```

Or using the older `torch.distributed.launch`:

```bash
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    train.py \
    --distributed \
    --world-size 4 \
    --epochs 20 \
    --batch-size 16
```

## Important Considerations

### Batch Size

The `--batch-size` argument specifies the **per-GPU batch size**. The effective batch size is:

```
Effective Batch Size = batch_size × num_gpus
```

For example, with `--batch-size 16` on 4 GPUs, the effective batch size is 64.

### Learning Rate

When using larger effective batch sizes, you may need to scale the learning rate accordingly. A common approach is linear scaling:

```
New LR = Base LR × (Effective Batch Size / Base Batch Size)
```

For example, if your base configuration uses `batch_size=16` and `lr=1e-4`, and you're now using 4 GPUs with `batch_size=16` per GPU:

```bash
# Effective batch size: 16 × 4 = 64
# Scaled LR: 1e-4 × (64 / 16) = 4e-4
./train_distributed.sh 4 --batch-size 16 --lr 4e-4
```

### Memory Usage

Each GPU will hold:
- A complete copy of the model
- A batch of data (size = batch_size)
- Gradients for the model parameters

Ensure each GPU has sufficient memory for the specified batch size.

### Checkpointing

Only the main process (rank 0) saves checkpoints to avoid conflicts. All processes participate in training, but logging and checkpoint saving are handled by rank 0.

### Data Loading

The implementation uses `DistributedSampler` to ensure each GPU sees different data. The sampler automatically:
- Partitions the dataset across GPUs
- Shuffles data independently each epoch
- Ensures no data duplication between GPUs

## Configuration Options

### Distributed Training Arguments

- `--distributed`: Enable distributed training mode
- `--world-size N`: Total number of processes (usually equals number of GPUs)
- `--local_rank N`: Local rank (automatically set by launcher, usually don't set manually)

### Environment Variables

The launcher scripts set these automatically:

- `MASTER_ADDR`: Address of the master node (default: 'localhost')
- `MASTER_PORT`: Port for process communication (default: '29500')
- `LOCAL_RANK`: Rank of the process on the current node
- `WORLD_SIZE`: Total number of processes

## Multi-Node Training

For training across multiple machines:

1. **On each node**, set the environment variables:

```bash
export MASTER_ADDR=<master_node_ip>
export MASTER_PORT=29500
```

2. **On the master node** (node 0):

```bash
torchrun \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr=<master_ip> \
    --master_port=29500 \
    train.py --distributed --world-size 8
```

3. **On worker nodes**:

```bash
torchrun \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr=<master_ip> \
    --master_port=29500 \
    train.py --distributed --world-size 8
```

## Troubleshooting

### NCCL Timeout Errors

If you encounter NCCL timeout errors:

```bash
export NCCL_TIMEOUT=1800  # Increase timeout to 30 minutes
export NCCL_DEBUG=INFO    # Enable debug logging
```

### Port Already in Use

If the default port (29500) is in use, specify a different one:

```bash
torchrun --master_port=29501 --nproc_per_node=4 train.py --distributed
```

### GPU Visibility

To train on specific GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,3 ./train_distributed.sh 3 --batch-size 16
```

### Out of Memory

If you run out of GPU memory:

1. Reduce the per-GPU batch size: `--batch-size 8`
2. Use a smaller model: `--backbone efficientnet_b3`
3. Reduce input size: `--input-size 384`
4. Enable gradient checkpointing (requires code modification)

## Performance Tips

1. **Use NCCL backend**: Automatically selected for CUDA devices (fastest for GPU)
2. **Pin memory**: Already enabled in the data loaders
3. **Use multiple workers**: `--num-workers 4` (adjust based on CPU cores)
4. **Mixed precision training**: Consider adding AMP for faster training (requires code modification)

## Verification

To verify distributed training is working correctly:

1. Check that all GPUs are being utilized:
   ```bash
   nvidia-smi  # Run in another terminal during training
   ```

2. Monitor the logs - you should see messages like:
   ```
   Distributed training initialized with N processes
   Using device: cuda:0 (rank 0)
   Model wrapped with DistributedDataParallel
   ```

3. Training speed should scale approximately linearly with GPU count

## Example Configurations

### Small Model, 2 GPUs
```bash
./train_distributed.sh 2 \
    --backbone efficientnet_b3 \
    --input-size 384 \
    --batch-size 32 \
    --lr 2e-4
```

### Large Model, 4 GPUs
```bash
./train_distributed.sh 4 \
    --backbone efficientnet_b5 \
    --input-size 512 \
    --batch-size 8 \
    --lr 4e-4
```

### Full Configuration, 8 GPUs
```bash
./train_distributed.sh 8 \
    --data-source s3 \
    --backbone efficientnet_b4 \
    --input-size 448 \
    --batch-size 16 \
    --epochs 50 \
    --lr 8e-4 \
    --weight-decay 1e-5 \
    --num-workers 8
```
