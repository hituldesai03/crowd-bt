"""
Train Crowd-BT model from pre-prepared training data.

This script loads pre-split train/val data and starts training.
"""

import argparse
import json
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from datetime import datetime

from config import Config
from model import QualityScorer
from loss import CrowdBTLoss
from dataset import create_data_loaders
from evaluate import full_evaluation, print_evaluation_report
from train import (
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    get_rank,
    train_epoch,
    validate,
    save_checkpoint
)


def main():
    parser = argparse.ArgumentParser(description='Train from prepared data')

    # Data args
    parser.add_argument('--data-dir', type=str, default='training_data',
                        help='Directory containing prepared training data')
    parser.add_argument('--image-dir', type=str,
                        default='/home/hitul/Desktop/quality-comparison-toolkit/data/iter_0',
                        help='Directory containing images')

    # Model args
    parser.add_argument('--backbone', type=str, default='efficientnet_b4',
                        help='Backbone architecture')
    parser.add_argument('--input-size', type=int, default=448,
                        help='Input image size')

    # Training args
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='Weight decay')

    # Crowd-BT args
    parser.add_argument('--eta-init', type=float, default=0.8,
                        help='Initial annotator reliability')
    parser.add_argument('--no-learnable-eta', action='store_true',
                        help='Disable learnable eta')

    # Other args
    parser.add_argument('--checkpoint-dir', type=str, default='results',
                        help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--no-auto-resume', action='store_true',
                        help='Disable automatic resume from last.pt')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')

    # Distributed training args
    parser.add_argument('--distributed', action='store_true',
                        help='Enable distributed training')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='Local rank for distributed training (automatically set by torch.distributed.launch)')
    parser.add_argument('--world-size', type=int, default=1,
                        help='Number of processes for distributed training')

    args = parser.parse_args()

    # Create config
    config = Config()

    # Update with command line args
    config.training.backbone_name = args.backbone
    config.training.input_size = args.input_size
    config.training.epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.lr
    config.training.weight_decay = args.weight_decay
    config.training.eta_init = args.eta_init
    config.training.eta_learnable = not args.no_learnable_eta
    config.training.checkpoint_dir = args.checkpoint_dir
    config.training.device = args.device
    config.training.num_workers = args.num_workers

    # Distributed training settings
    config.training.distributed = args.distributed
    if 'LOCAL_RANK' in os.environ:
        config.training.local_rank = int(os.environ['LOCAL_RANK'])
    else:
        config.training.local_rank = args.local_rank
    config.training.world_size = args.world_size

    config.data.image_dir = args.image_dir

    # Initialize distributed training if enabled
    if config.training.distributed:
        rank = config.training.local_rank
        world_size = config.training.world_size

        backend = 'nccl' if config.training.device == 'cuda' else 'gloo'
        setup_distributed(rank, world_size, backend)

        if config.training.device == "cuda":
            device = f"cuda:{rank}"
            torch.cuda.set_device(rank)
        else:
            device = "cpu"

        if is_main_process():
            print(f"Distributed training initialized with {world_size} processes")
            print(f"Using device: {device} (rank {rank})")
    else:
        device = config.training.device
        if device == "cuda" and not torch.cuda.is_available():
            if is_main_process():
                print("CUDA not available, using CPU")
            device = "cpu"
        if is_main_process():
            print(f"Using device: {device}")

    # Load metadata to show what we're training on
    metadata_path = os.path.join(args.data_dir, 'metadata.json')
    metadata = {}
    uses_fixed_eta = False
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        uses_fixed_eta = metadata.get('uses_fixed_eta', False)
        if is_main_process():
            print("="*60)
            print("Training Data Info")
            print("="*60)
            print(f"Data prepared: {metadata.get('created_at', 'unknown')}")
            print(f"Mode: {metadata.get('mode', 'unknown')}")
            print(f"Uses fixed eta: {uses_fixed_eta}")
            print(f"Number of users: {metadata.get('num_users', 'unknown')}")
            print(f"Avg user reliability: {metadata.get('avg_user_reliability', 0):.3f}")
            print(f"Total samples: {metadata.get('total_samples', 0)}")
            print(f"Train samples: {metadata.get('train_samples', 0)}")
            print(f"Val samples: {metadata.get('val_samples', 0)}")
            if config.training.distributed:
                print(f"Batch size per GPU: {config.training.batch_size}")
                print(f"Effective batch size: {config.training.batch_size * world_size}")
            print()

    # Load train/val data
    train_path = os.path.join(args.data_dir, 'train_comparisons.json')
    val_path = os.path.join(args.data_dir, 'val_comparisons.json')

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        if is_main_process():
            print(f"Error: Training data not found in {args.data_dir}")
            print("Please run prepare_s3_data.py first")
        if config.training.distributed:
            cleanup_distributed()
        return

    if is_main_process():
        print(f"Loading training data from {args.data_dir}...")

    with open(train_path, 'r') as f:
        train_data = json.load(f)

    with open(val_path, 'r') as f:
        val_data = json.load(f)

    if is_main_process():
        print(f"Loaded {len(train_data)} train and {len(val_data)} val samples")

    # Save config
    if is_main_process():
        config_path = os.path.join(args.checkpoint_dir, 'config.yaml')
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        config.to_yaml(config_path)
        print(f"Config saved to {config_path}")
    else:
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Create data loaders
    if is_main_process():
        print("\nCreating data loaders...")
    train_loader, val_loader = create_data_loaders(
        train_data,
        val_data,
        image_dir=config.data.image_dir,
        input_size=config.training.input_size,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        distributed=config.training.distributed
    )

    if is_main_process():
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Create model
    if is_main_process():
        print("\nCreating model...")
    model = QualityScorer(
        backbone_name=config.training.backbone_name,
        pretrained=config.training.pretrained
    )
    model = model.to(device)

    # Wrap model with DistributedDataParallel if using distributed training
    if config.training.distributed:
        model = DDP(
            model,
            device_ids=[config.training.local_rank] if device.startswith('cuda') else None,
            output_device=config.training.local_rank if device.startswith('cuda') else None
        )
        if is_main_process():
            print("Model wrapped with DistributedDataParallel")

    # Create loss function
    # If using fixed eta (individual annotations), eta is not learnable
    if is_main_process():
        if uses_fixed_eta:
            print("Using per-annotation reliability as fixed eta (correct Crowd-BT)")
        else:
            print(f"Using global {'learnable' if config.training.eta_learnable else 'fixed'} eta")

    criterion = CrowdBTLoss(
        eta_init=config.training.eta_init,
        eta_learnable=config.training.eta_learnable and not uses_fixed_eta
    )
    criterion = criterion.to(device)

    # Optimizer - only include eta if learnable and not using fixed eta
    params = list(model.parameters())
    if config.training.eta_learnable and not uses_fixed_eta:
        params.append(criterion.eta_logit)

    optimizer = AdamW(
        params,
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )

    # Learning rate scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.training.epochs,
        eta_min=config.training.learning_rate * 0.01
    )

    # Resume from checkpoint
    start_epoch = 0
    best_val_acc = 0.0

    # Auto-resume: look for last.pt if not disabled
    checkpoint_path = args.resume
    if checkpoint_path is None and not args.no_auto_resume:
        last_checkpoint = os.path.join(args.checkpoint_dir, 'last.pt')
        if os.path.exists(last_checkpoint):
            checkpoint_path = last_checkpoint
            if is_main_process():
                print(f"\nFound existing checkpoint: {checkpoint_path}")
                print("Auto-resuming training (use --no-auto-resume to disable)")

    if checkpoint_path and os.path.exists(checkpoint_path):
        if is_main_process():
            print(f"Resuming from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Get the actual model (unwrap DDP if necessary)
        model_to_load = model.module if isinstance(model, DDP) else model
        model_to_load.load_state_dict(checkpoint['model'])

        optimizer.load_state_dict(checkpoint['optimizer'])
        if checkpoint['scheduler']:
            scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint['metrics'].get('val_accuracy', 0.0)

        # Load global eta parameter if present
        if 'eta_logit' in checkpoint and hasattr(criterion, 'eta_logit'):
            criterion.eta_logit.data = torch.tensor(checkpoint['eta_logit'])

        if is_main_process():
            print(f"Resumed from epoch {start_epoch} with best val acc: {best_val_acc:.3f}")

    # Training loop
    if is_main_process():
        print("\n" + "="*60)
        print("Starting Training")
        print("="*60)

    training_log = []

    for epoch in range(start_epoch, config.training.epochs):
        # Set epoch for DistributedSampler to ensure proper shuffling
        if config.training.distributed:
            train_loader.sampler.set_epoch(epoch)

        if is_main_process():
            print(f"\n{'='*50}")
            print(f"Epoch {epoch + 1}/{config.training.epochs}")
            print(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")
            if hasattr(criterion, 'eta'):
                print(f"Eta (annotator reliability): {criterion.eta.item():.4f}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device,
            log_interval=config.training.log_interval
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step()

        # Log (only on main process)
        if is_main_process():
            epoch_log = {
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'train_accuracy': train_metrics['accuracy'],
                'val_loss': val_metrics['loss'],
                'val_accuracy': val_metrics['accuracy'],
                'learning_rate': scheduler.get_last_lr()[0],
                'eta': criterion.eta.item() if hasattr(criterion, 'eta') else None,
                'timestamp': datetime.now().isoformat()
            }
            training_log.append(epoch_log)

            print(f"\nTrain Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.3f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.3f}")

            # Save best model
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                best_path = os.path.join(config.training.checkpoint_dir, 'best_model.pt')
                save_checkpoint(
                    model, optimizer, scheduler, criterion,
                    epoch, {'val_accuracy': val_metrics['accuracy'], **val_metrics},
                    best_path, config, is_best=True
                )
                print(f"New best model! Val Acc: {best_val_acc:.3f}")

            # Save latest checkpoint after every epoch for auto-resume
            last_path = os.path.join(config.training.checkpoint_dir, 'last.pt')
            save_checkpoint(
                model, optimizer, scheduler, criterion,
                epoch, {'val_accuracy': val_metrics['accuracy'], **val_metrics},
                last_path, config, is_best=False
            )

            # Save periodic checkpoint
            if (epoch + 1) % config.training.save_every == 0:
                ckpt_path = os.path.join(
                    config.training.checkpoint_dir,
                    f'checkpoint_epoch_{epoch + 1}.pt'
                )
                save_checkpoint(
                    model, optimizer, scheduler, criterion,
                    epoch, {'val_accuracy': val_metrics['accuracy'], **val_metrics},
                    ckpt_path, config, is_best=False
                )

            # Save training log
            log_path = os.path.join(config.training.checkpoint_dir, 'training_log.json')
            with open(log_path, 'w') as f:
                json.dump(training_log, f, indent=2)

        # Synchronize all processes before continuing to next epoch
        if config.training.distributed:
            dist.barrier()

    # Save final model (only on main process)
    if is_main_process():
        final_path = os.path.join(config.training.checkpoint_dir, 'final_model.pt')
        save_checkpoint(
            model, optimizer, scheduler, criterion,
            config.training.epochs - 1,
            {'val_accuracy': val_metrics['accuracy'], **val_metrics},
            final_path, config
        )

        print(f"\nTraining complete!")
        print(f"Best validation accuracy: {best_val_acc:.3f}")
        print(f"Models saved to: {config.training.checkpoint_dir}")

    # Synchronize before evaluation
    if config.training.distributed:
        dist.barrier()

    # Run evaluation (only on main process)
    if is_main_process():
        print("\n" + "=" * 50)
        print("Running Full Evaluation on Best Model")
        print("=" * 50)

        best_model_path = os.path.join(config.training.checkpoint_dir, 'best_model.pt')
        if os.path.exists(best_model_path):
            from model import load_model

            eval_model = load_model(
                best_model_path,
                backbone_name=config.training.backbone_name,
                device=device
            )

            # Combine train and val data for full evaluation
            eval_comparisons = train_data + val_data

            eval_results = full_evaluation(
                eval_model,
                config,
                comparisons=eval_comparisons,
                device=device
            )

            print_evaluation_report(eval_results)

            # Save evaluation results
            eval_results_path = os.path.join(config.training.checkpoint_dir, 'evaluation_results.json')
            with open(eval_results_path, 'w') as f:
                json.dump(eval_results, f, indent=2)
            print(f"\nFull evaluation results saved to {eval_results_path}")

    # Clean up distributed training
    if config.training.distributed:
        cleanup_distributed()


if __name__ == '__main__':
    main()
