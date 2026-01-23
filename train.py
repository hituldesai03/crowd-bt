"""
Training script for Crowd-BT Quality Scorer.
"""

import argparse
import os
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from datetime import datetime
from typing import Dict, Optional

from config import Config, default_config
from model import QualityScorer
from loss import CrowdBTLoss, get_loss_function
from dataset import (
    PairwiseComparisonDataset,
    create_data_loaders,
    split_comparisons
)
from data_loader import (
    load_all_comparisons,
    load_local_comparisons,
    prepare_training_data
)
from evaluate import full_evaluation, print_evaluation_report


def train_epoch(
    model: nn.Module,
    train_loader,
    criterion: nn.Module,
    optimizer,
    device: str,
    log_interval: int = 10
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, batch in enumerate(pbar):
        img1 = batch['img1'].to(device)
        img2 = batch['img2'].to(device)
        labels = batch['label'].to(device)
        weights = batch['weight'].to(device)
        annotator_ids = batch.get('annotator_id', None)

        optimizer.zero_grad()

        # Forward pass
        score1 = model(img1)
        score2 = model(img2)
        score_diff = score1 - score2

        # Compute loss (pass annotator_ids for per-user eta)
        loss_dict = criterion(score_diff, labels, weights, annotator_ids)
        loss = loss_dict['loss']

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += loss_dict['accuracy'].item()
        num_batches += 1

        if batch_idx % log_interval == 0:
            eta_str = f", eta={loss_dict['eta']:.3f}" if 'eta' in loss_dict else ""
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{loss_dict['accuracy'].item():.3f}{eta_str}"
            })

    return {
        'loss': total_loss / num_batches,
        'accuracy': total_acc / num_batches
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader,
    criterion: nn.Module,
    device: str
) -> Dict[str, float]:
    """Validate the model."""
    model.eval()

    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0

    for batch in tqdm(val_loader, desc="Validating"):
        img1 = batch['img1'].to(device)
        img2 = batch['img2'].to(device)
        labels = batch['label'].to(device)
        weights = batch['weight'].to(device)
        annotator_ids = batch.get('annotator_id', None)

        score1 = model(img1)
        score2 = model(img2)
        score_diff = score1 - score2

        loss_dict = criterion(score_diff, labels, weights, annotator_ids)

        total_loss += loss_dict['loss'].item()
        total_acc += loss_dict['accuracy'].item()
        num_batches += 1

    return {
        'loss': total_loss / num_batches,
        'accuracy': total_acc / num_batches
    }


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Find the latest checkpoint in the checkpoint directory.

    Args:
        checkpoint_dir: Directory to search for checkpoints

    Returns:
        Path to the latest checkpoint (last.pt), or None if not found
    """
    last_checkpoint = os.path.join(checkpoint_dir, 'last.pt')
    if os.path.exists(last_checkpoint):
        return last_checkpoint
    return None


def save_checkpoint(
    model: nn.Module,
    optimizer,
    scheduler,
    criterion: nn.Module,
    epoch: int,
    metrics: Dict,
    path: str,
    config: Config,
    is_best: bool = False
):
    """Save a training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'config': {
            'backbone_name': config.training.backbone_name,
            'input_size': config.training.input_size,
        }
    }

    # Save eta parameters if using Crowd-BT loss
    if hasattr(criterion, 'use_per_user_eta') and criterion.use_per_user_eta:
        # Save per-user eta parameters
        user_eta_dict = {}
        if criterion.eta_learnable:
            for user_id, param in criterion.user_eta_logits.items():
                user_eta_dict[user_id] = param.data.item()
            checkpoint['user_eta_logits'] = user_eta_dict
            checkpoint['default_eta_logit'] = criterion.default_eta_logit.data.item()
        else:
            # Non-learnable etas (stored as buffers)
            for name, buffer in criterion.named_buffers():
                if name.startswith('eta_logit_'):
                    user_id = name[len('eta_logit_'):]
                    user_eta_dict[user_id] = buffer.item()
            checkpoint['user_eta_logits'] = user_eta_dict
            if hasattr(criterion, 'default_eta_logit'):
                checkpoint['default_eta_logit'] = criterion.default_eta_logit.item()
        checkpoint['use_per_user_eta'] = True
    elif hasattr(criterion, 'eta_logit'):
        # Save single global eta
        checkpoint['eta_logit'] = criterion.eta_logit.data.item()
        checkpoint['use_per_user_eta'] = False

    torch.save(checkpoint, path)

    if is_best:
        print(f"Saved best checkpoint to {path}")
    else:
        print(f"Saved checkpoint to {path}")


def train(
    config: Config = None,
    data_source: str = "local",
    checkpoint_path: Optional[str] = None,
    auto_resume: bool = True
):
    """
    Main training function.

    Args:
        config: Training configuration
        data_source: 'local' or 's3'
        checkpoint_path: Optional path to resume from. If None and auto_resume is True,
                        will automatically resume from last.pt if it exists
        auto_resume: Whether to automatically resume from last.pt if checkpoint_path is None
    """
    if config is None:
        config = default_config

    device = config.training.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"

    print(f"Using device: {device}")
    print(f"Backbone: {config.training.backbone_name}")
    print(f"Input size: {config.training.input_size}")

    # Load comparison data
    print("\nLoading comparison data...")
    user_reliabilities = None  # Will be populated if loading from S3
    if data_source == "s3":
        comparisons, user_reliabilities = load_all_comparisons(
            bucket_name=config.data.bucket_name,
            data_prefix=config.data.data_prefix,
            min_reliability=0.3,
            exclude_dummy=True
        )
        training_data = prepare_training_data(comparisons, use_gold_labels=True)
    else:
        # Load from local files
        local_comparisons = load_local_comparisons(data_dir=config.data.local_data_dir)

        # Convert to training format
        training_data = []
        for comp in local_comparisons:
            label = 0
            expected = comp.get('expected_choice')
            if expected == 'left':
                label = 1
            elif expected == 'right':
                label = -1

            training_data.append({
                'img1': comp['img1'],
                'img2': comp['img2'],
                'label': label,
                'weight': 1.0,
                'pair_type': comp.get('type', 'unknown'),
            })

    print(f"Total training samples: {len(training_data)}")

    # Split data
    train_data, val_data = split_comparisons(
        training_data,
        train_ratio=config.training.train_split,
        stratify_by_pair_type=True
    )

    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_data,
        val_data,
        image_dir=config.data.image_dir,
        input_size=config.training.input_size,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers
    )

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Create model
    model = QualityScorer(
        backbone_name=config.training.backbone_name,
        pretrained=config.training.pretrained
    )
    model = model.to(device)

    # Create loss function with per-user eta initialization
    criterion = CrowdBTLoss(
        eta_init=config.training.eta_init,
        eta_learnable=config.training.eta_learnable,
        user_reliabilities=user_reliabilities  # Per-user eta if available
    )
    criterion = criterion.to(device)

    # Print eta initialization info
    if criterion.use_per_user_eta:
        print(f"\nInitialized per-user eta for {len(user_reliabilities)} users")
        print(f"Per-user eta values: min={min(user_reliabilities.values()):.3f}, "
              f"max={max(user_reliabilities.values()):.3f}, "
              f"mean={sum(user_reliabilities.values())/len(user_reliabilities):.3f}")
    else:
        print(f"\nUsing global eta initialized to {config.training.eta_init:.3f}")

    # Optimizer - include all eta parameters if learnable
    params = list(model.parameters())
    if config.training.eta_learnable:
        if criterion.use_per_user_eta:
            # Add all per-user eta parameters
            params.extend(criterion.user_eta_logits.parameters())
            params.append(criterion.default_eta_logit)
        else:
            # Add single global eta parameter
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

    # Auto-resume: if no checkpoint_path provided and auto_resume enabled, look for last.pt
    if checkpoint_path is None and auto_resume:
        checkpoint_path = find_latest_checkpoint(config.training.checkpoint_dir)
        if checkpoint_path:
            print(f"\nFound existing checkpoint: {checkpoint_path}")
            print("Auto-resuming training (use --no-auto-resume to disable)")
        else:
            print("\nNo checkpoint found, starting training from scratch")
    elif checkpoint_path is None:
        print("\nStarting training from scratch (auto-resume disabled)")

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Resuming from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if checkpoint['scheduler']:
            scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint['metrics'].get('val_accuracy', 0.0)

        # Load eta parameters
        if checkpoint.get('use_per_user_eta', False):
            # Load per-user eta parameters
            if 'user_eta_logits' in checkpoint:
                for user_id, eta_value in checkpoint['user_eta_logits'].items():
                    if criterion.eta_learnable and user_id in criterion.user_eta_logits:
                        criterion.user_eta_logits[user_id].data = torch.tensor(eta_value)
                    elif hasattr(criterion, f'eta_logit_{user_id}'):
                        getattr(criterion, f'eta_logit_{user_id}').data = torch.tensor(eta_value)
            if 'default_eta_logit' in checkpoint:
                criterion.default_eta_logit.data = torch.tensor(checkpoint['default_eta_logit'])
            print(f"Loaded per-user eta parameters for {len(checkpoint.get('user_eta_logits', {}))} users")
        elif 'eta_logit' in checkpoint:
            # Load single global eta (backward compatible)
            if hasattr(criterion, 'eta_logit'):
                criterion.eta_logit.data = torch.tensor(checkpoint['eta_logit'])

        print(f"Resumed from epoch {start_epoch} with best val acc: {best_val_acc:.3f}")

    # Create checkpoint directory
    os.makedirs(config.training.checkpoint_dir, exist_ok=True)

    # Training log
    log_path = os.path.join(config.training.checkpoint_dir, 'training_log.json')
    training_log = []

    # Training loop
    print("\nStarting training...")
    for epoch in range(start_epoch, config.training.epochs):
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

        # Log
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
        with open(log_path, 'w') as f:
            json.dump(training_log, f, indent=2)

    # Save final model
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

    # Run full evaluation on the best model
    print("\n" + "=" * 50)
    print("Running Full Evaluation on Best Model")
    print("=" * 50)

    # Load the best model for evaluation
    best_model_path = os.path.join(config.training.checkpoint_dir, 'best_model.pt')
    if os.path.exists(best_model_path):
        from model import load_model
        eval_model = load_model(
            best_model_path,
            backbone_name=config.training.backbone_name,
            device=device
        )

        # Load comparisons for evaluation
        if data_source == "local":
            eval_comparisons = load_local_comparisons(data_dir=config.data.local_data_dir)
        else:
            eval_comparisons = training_data

        # Run evaluation
        eval_results = full_evaluation(
            eval_model,
            config,
            comparisons=eval_comparisons,
            device=device
        )

        # Print evaluation report
        print_evaluation_report(eval_results)

        # Add evaluation results to training log
        eval_log_entry = {
            'evaluation': {
                'overall_accuracy': eval_results.get('overall_accuracy'),
                'kendall_tau': eval_results.get('kendall_tau', {}).get('tau'),
                'spearman_rho': eval_results.get('spearman_rho', {}).get('rho'),
                'gold_hierarchy_preserved': eval_results.get('gold_hierarchy', {}).get('ordering_preserved'),
                'gold_hierarchy_violations': eval_results.get('gold_hierarchy', {}).get('violation_count', 0),
                'accuracy_by_pair_type': {
                    k: v.get('accuracy') for k, v in eval_results.get('accuracy_by_pair_type', {}).items()
                }
            },
            'timestamp': datetime.now().isoformat()
        }
        training_log.append(eval_log_entry)

        # Save updated training log with evaluation results
        with open(log_path, 'w') as f:
            json.dump(training_log, f, indent=2)

        # Save full evaluation results separately
        eval_results_path = os.path.join(config.training.checkpoint_dir, 'evaluation_results.json')
        with open(eval_results_path, 'w') as f:
            json.dump(eval_results, f, indent=2)
        print(f"\nFull evaluation results saved to {eval_results_path}")
    else:
        print(f"Warning: Best model not found at {best_model_path}, skipping evaluation")

    return model, training_log


def main():
    parser = argparse.ArgumentParser(description='Train Crowd-BT Quality Scorer')

    # Model args
    parser.add_argument('--backbone', type=str, default='efficientnet_b4',
                        help='Backbone architecture')
    parser.add_argument('--input-size', type=int, default=448,
                        help='Input image size')

    # Training args
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
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

    # Data args
    parser.add_argument('--data-source', type=str, default='local',
                        choices=['local', 's3'],
                        help='Data source')
    parser.add_argument('--image-dir', type=str,
                        default='/home/hitul/Desktop/quality-comparison-toolkit/data/iter_0',
                        help='Directory containing images')
    parser.add_argument('--data-dir', type=str,
                        default='/home/hitul/Desktop/quality-comparison-toolkit/data',
                        help='Directory containing comparison JSON files')

    # Other args
    parser.add_argument('--checkpoint-dir', type=str, default='results',
                        help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from specific checkpoint path')
    parser.add_argument('--no-auto-resume', action='store_true',
                        help='Disable automatic resume from last.pt')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')

    # Config file
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file')

    args = parser.parse_args()

    # Load config
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = Config()

    # Override with command line args
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

    config.data.image_dir = args.image_dir
    config.data.local_data_dir = args.data_dir

    # Save config
    config_path = os.path.join(args.checkpoint_dir, 'config.yaml')
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    config.to_yaml(config_path)
    print(f"Config saved to {config_path}")

    # Train
    train(
        config,
        data_source=args.data_source,
        checkpoint_path=args.resume,
        auto_resume=not args.no_auto_resume
    )


if __name__ == '__main__':
    main()
