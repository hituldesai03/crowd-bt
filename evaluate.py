"""
Evaluation metrics for Crowd-BT Quality Scorer.

Provides evaluation based on pairwise comparison accuracy.
Gold-gold pairs are used as the reliable ground truth.
"""

import argparse
import os
import json
import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
from collections import defaultdict
from tqdm import tqdm

from config import Config, default_config
from model import QualityScorer, load_model
from dataset import get_transforms
from data_loader import load_local_comparisons


def load_prepared_data_comparisons(
    prepared_data_dir: str,
    split: str = "val"
) -> List[Dict]:
    """
    Load comparisons from a prepared data directory (e.g., training_data_120k).

    Args:
        prepared_data_dir: Path to directory containing train_comparisons.json,
                          val_comparisons.json, all_comparisons.json
        split: Which split to load - "train", "val", or "all"

    Returns:
        List of comparison dicts with keys: img1, img2, label, weight,
        annotator_id, annotator_reliability, pair_type, img1_category, img2_category
    """
    split_files = {
        "train": "train_comparisons.json",
        "val": "val_comparisons.json",
        "all": "all_comparisons.json"
    }

    if split not in split_files:
        raise ValueError(f"Invalid split '{split}'. Must be one of: {list(split_files.keys())}")

    json_path = os.path.join(prepared_data_dir, split_files[split])

    if not os.path.exists(json_path):
        print(f"Warning: {json_path} not found")
        return []

    with open(json_path, 'r') as f:
        comparisons = json.load(f)

    print(f"Loaded {len(comparisons)} comparisons from {json_path}")
    return comparisons


@torch.no_grad()
def evaluate_pairwise_accuracy(
    model: QualityScorer,
    dataloader: DataLoader,
    device: str = "cuda"
) -> Dict[str, float]:
    """
    Compute pairwise prediction accuracy from a dataloader.

    Args:
        model: Trained QualityScorer model
        dataloader: DataLoader with pairwise comparisons
        device: Device to run on

    Returns:
        Dict with accuracy metrics
    """
    model.eval()
    correct = 0
    total = 0

    for batch in tqdm(dataloader, desc="Computing pairwise accuracy"):
        img1 = batch['img1'].to(device)
        img2 = batch['img2'].to(device)
        labels = batch['label'].to(device)

        score1 = model(img1)
        score2 = model(img2)
        score_diff = (score1 - score2).squeeze()

        # Predict: positive diff -> img1 wins (1), negative -> img2 wins (-1)
        predictions = torch.sign(score_diff)

        # Only count non-draw pairs for accuracy
        non_draw_mask = labels != 0
        if non_draw_mask.any():
            correct += (predictions[non_draw_mask] == labels[non_draw_mask]).sum().item()
            total += non_draw_mask.sum().item()

    accuracy = correct / total if total > 0 else 0.0
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total
    }


@torch.no_grad()
def evaluate_by_pair_type(
    model: QualityScorer,
    comparisons: List[Dict],
    image_dir: str,
    input_size: int = 448,
    device: str = "cuda"
) -> Dict[str, Dict]:
    """
    Compute accuracy breakdown by pair type.

    Gold-gold pairs provide the most reliable ground truth.

    Args:
        model: Trained model
        comparisons: List of comparison dicts
        image_dir: Directory containing images
        input_size: Model input size
        device: Device

    Returns:
        Dict mapping pair_type to accuracy metrics
    """
    from PIL import Image

    model.eval()
    transform = get_transforms(input_size, is_train=False)

    # Group comparisons by pair type
    by_type = defaultdict(list)
    for comp in comparisons:
        pair_type = comp.get('pair_type', comp.get('type', 'unknown'))
        by_type[pair_type].append(comp)

    results = {}

    for pair_type, type_comps in by_type.items():
        correct = 0
        total = 0
        score_diffs = []

        for comp in tqdm(type_comps, desc=f"Evaluating {pair_type}", leave=False):
            img1_path = os.path.join(image_dir, comp['img1'])
            img2_path = os.path.join(image_dir, comp['img2'])

            if not os.path.exists(img1_path) or not os.path.exists(img2_path):
                continue

            try:
                img1 = Image.open(img1_path).convert('RGB')
                img2 = Image.open(img2_path).convert('RGB')

                img1_tensor = transform(img1).unsqueeze(0).to(device)
                img2_tensor = transform(img2).unsqueeze(0).to(device)

                score1 = model(img1_tensor).item()
                score2 = model(img2_tensor).item()
                score_diff = score1 - score2

                # Get expected label
                label = comp.get('label')
                if label is None:
                    expected = comp.get('expected_choice')
                    if expected == 'left':
                        label = 1
                    elif expected == 'right':
                        label = -1
                    else:
                        label = 0

                # Skip draws for accuracy calculation
                if label == 0:
                    continue

                # Predict based on score difference
                if score_diff > 0:
                    pred = 1
                elif score_diff < 0:
                    pred = -1
                else:
                    pred = 0

                score_diffs.append(abs(score_diff))

                if pred == label:
                    correct += 1
                total += 1

            except Exception as e:
                print(f"Error processing comparison: {e}")
                continue

        accuracy = correct / total if total > 0 else 0.0
        mean_score_diff = sum(score_diffs) / len(score_diffs) if score_diffs else 0.0

        results[pair_type] = {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'num_comparisons': len(type_comps),
            'mean_score_margin': mean_score_diff
        }

    return results


@torch.no_grad()
def evaluate_gold_pairs(
    model: QualityScorer,
    comparisons: List[Dict],
    image_dir: str,
    input_size: int = 448,
    device: str = "cuda"
) -> Dict:
    """
    Evaluate specifically on gold-gold pairs (most reliable ground truth).

    Args:
        model: Trained model
        comparisons: List of comparison dicts
        image_dir: Directory containing images
        input_size: Model input size
        device: Device

    Returns:
        Dict with gold pair evaluation results
    """
    # Filter to gold-gold pairs only
    gold_pairs = [
        c for c in comparisons
        if c.get('pair_type', c.get('type', '')) == 'gold_gold'
    ]

    if not gold_pairs:
        return {
            'accuracy': None,
            'correct': 0,
            'total': 0,
            'message': 'No gold-gold pairs found'
        }

    # Use evaluate_by_pair_type but only for gold_gold
    results = evaluate_by_pair_type(
        model, gold_pairs, image_dir, input_size, device
    )

    return results.get('gold_gold', {
        'accuracy': None,
        'correct': 0,
        'total': 0
    })


def full_evaluation(
    model: QualityScorer,
    config: Config,
    comparisons: Optional[List[Dict]] = None,
    device: str = "cuda"
) -> Dict:
    """
    Run all evaluation metrics.

    Gold-gold pairs are the primary reliable ground truth.

    Args:
        model: Trained model
        config: Configuration object
        comparisons: Optional list of comparisons (loaded from config if not provided)
        device: Device to run on

    Returns:
        Dict with all evaluation results
    """
    results = {}

    # Load comparisons if not provided
    if comparisons is None:
        comparisons = load_local_comparisons(data_dir=config.data.local_data_dir)

    if not comparisons:
        print("No comparisons found for evaluation")
        return results

    image_dir = config.data.image_dir
    input_size = config.training.input_size

    print(f"\nTotal comparisons: {len(comparisons)}")

    # 1. Accuracy by pair type
    print("\n=== Evaluating by Pair Type ===")
    pair_type_results = evaluate_by_pair_type(
        model, comparisons, image_dir, input_size, device
    )
    results['accuracy_by_pair_type'] = pair_type_results

    # 2. Compute overall accuracy (across all pair types)
    total_correct = sum(r['correct'] for r in pair_type_results.values())
    total_pairs = sum(r['total'] for r in pair_type_results.values())
    results['overall_accuracy'] = total_correct / total_pairs if total_pairs > 0 else 0.0
    results['overall_correct'] = total_correct
    results['overall_total'] = total_pairs

    # 3. Gold-gold accuracy (primary reliable metric)
    if 'gold_gold' in pair_type_results:
        results['gold_accuracy'] = pair_type_results['gold_gold']['accuracy']
        results['gold_correct'] = pair_type_results['gold_gold']['correct']
        results['gold_total'] = pair_type_results['gold_gold']['total']
    else:
        results['gold_accuracy'] = None
        results['gold_correct'] = 0
        results['gold_total'] = 0

    return results


def print_evaluation_report(results: Dict) -> None:
    """
    Print a formatted evaluation report.

    Args:
        results: Dict from full_evaluation()
    """
    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)

    # Gold accuracy (primary metric)
    if results.get('gold_accuracy') is not None:
        gold_acc = results['gold_accuracy'] * 100
        print(f"\nGold-Gold Pair Accuracy (Primary GT): {gold_acc:.1f}%")
        print(f"  ({results['gold_correct']}/{results['gold_total']} correct)")
    else:
        print("\nGold-Gold Pair Accuracy: N/A (no gold pairs found)")

    # Overall accuracy
    if 'overall_accuracy' in results:
        overall_acc = results['overall_accuracy'] * 100
        print(f"\nOverall Pairwise Accuracy: {overall_acc:.1f}%")
        print(f"  ({results['overall_correct']}/{results['overall_total']} correct)")

    # Accuracy by pair type
    if 'accuracy_by_pair_type' in results:
        print("\n--- Accuracy by Pair Type ---")
        for pair_type, metrics in sorted(results['accuracy_by_pair_type'].items()):
            acc = metrics['accuracy'] * 100
            margin = metrics.get('mean_score_margin', 0)
            print(f"  {pair_type:20s}: {acc:5.1f}% "
                  f"({metrics['correct']:6d}/{metrics['total']:6d}) "
                  f"margin={margin:.3f}")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Evaluate Crowd-BT Quality Scorer')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--backbone', type=str, default='efficientnet_b4',
                        help='Backbone architecture')
    parser.add_argument('--input-size', type=int, default=448,
                        help='Input image size')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')

    # Data args
    parser.add_argument('--image-dir', type=str,
                        default='/home/hitul/Desktop/quality-comparison-toolkit/data/iter_0',
                        help='Directory containing images')
    parser.add_argument('--data-dir', type=str,
                        default='/home/hitul/Desktop/quality-comparison-toolkit/data',
                        help='Directory containing comparison JSON files')

    # Prepared data format (e.g., training_data_120k)
    parser.add_argument('--prepared-data-dir', type=str, default=None,
                        help='Path to prepared data directory (e.g., training_data_120k) '
                             'containing train_comparisons.json, val_comparisons.json, etc.')
    parser.add_argument('--split', type=str, default='val',
                        choices=['train', 'val', 'all'],
                        help='Which split to evaluate when using --prepared-data-dir')

    # Output
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for results')

    args = parser.parse_args()

    # Setup device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"

    # Load model
    print(f"Loading model from {args.checkpoint}")

    # Try to read backbone from checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if 'config' in checkpoint:
        backbone = checkpoint['config'].get('backbone_name', args.backbone)
        input_size = checkpoint['config'].get('input_size', args.input_size)
    else:
        backbone = args.backbone
        input_size = args.input_size

    model = load_model(args.checkpoint, backbone_name=backbone, device=device)
    print(f"Model loaded: {backbone}, input size: {input_size}")

    # Load comparisons based on data source
    if args.prepared_data_dir:
        # Load from prepared data directory (training_data_120k format)
        comparisons = load_prepared_data_comparisons(
            args.prepared_data_dir,
            split=args.split
        )
        image_dir = args.image_dir
    else:
        # Load from legacy data directory
        comparisons = load_local_comparisons(data_dir=args.data_dir)
        image_dir = args.image_dir

    if not comparisons:
        print("No comparisons found for evaluation")
        return

    print(f"\nTotal comparisons: {len(comparisons)}")
    print(f"Image directory: {image_dir}")

    # Run evaluation
    print("\n=== Evaluating by Pair Type ===")
    pair_type_results = evaluate_by_pair_type(
        model, comparisons, image_dir, input_size, device
    )

    # Compile results
    results = {'accuracy_by_pair_type': pair_type_results}

    # Compute overall accuracy
    total_correct = sum(r['correct'] for r in pair_type_results.values())
    total_pairs = sum(r['total'] for r in pair_type_results.values())
    results['overall_accuracy'] = total_correct / total_pairs if total_pairs > 0 else 0.0
    results['overall_correct'] = total_correct
    results['overall_total'] = total_pairs

    # Gold-gold accuracy
    if 'gold_gold' in pair_type_results:
        results['gold_accuracy'] = pair_type_results['gold_gold']['accuracy']
        results['gold_correct'] = pair_type_results['gold_gold']['correct']
        results['gold_total'] = pair_type_results['gold_gold']['total']
    else:
        results['gold_accuracy'] = None
        results['gold_correct'] = 0
        results['gold_total'] = 0

    # Print report
    print_evaluation_report(results)

    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
