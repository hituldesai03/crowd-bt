"""
Prepare golden_ranking training data for crowd-bt model training.

This script loads golden_ranking comparisons from the quality-comparison-toolkit
and prepares them for training with proper train/val splits.
"""

import json
import os
import argparse
from typing import List, Dict
from datetime import datetime
from config import get_quality_rank


def load_golden_ranking_comparisons(comparison_file: str) -> List[Dict]:
    """
    Load golden_ranking comparison data.

    Args:
        comparison_file: Path to gold_ranking_comparisons.json

    Returns:
        List of comparison dicts
    """
    with open(comparison_file, 'r') as f:
        comparisons = json.load(f)

    print(f"Loaded {len(comparisons)} golden_ranking comparisons")
    return comparisons


def add_ground_truth_labels(comparisons: List[Dict]) -> List[Dict]:
    """
    Add ground truth labels based on quality rankings.

    Args:
        comparisons: List of comparison dicts with img1_category and img2_category

    Returns:
        List of comparisons with added 'label' and 'expected_choice' fields
    """
    labeled_comparisons = []

    for comp in comparisons:
        img1_cat = comp.get('img1_category', '')
        img2_cat = comp.get('img2_category', '')

        rank1 = get_quality_rank(img1_cat)
        rank2 = get_quality_rank(img2_cat)

        # Determine label: 1 if img1 > img2, -1 if img1 < img2, 0 if equal
        if rank1 > rank2:
            label = 1
            expected = 'left'
        elif rank1 < rank2:
            label = -1
            expected = 'right'
        else:
            label = 0
            expected = 'draw'

        comp['label'] = label
        comp['expected_choice'] = expected
        comp['weight'] = 1.0  # All golden_ranking pairs are reliable
        comp['pair_type'] = 'gold_ranking'

        labeled_comparisons.append(comp)

    return labeled_comparisons


def split_data(comparisons: List[Dict], train_ratio: float = 0.8) -> tuple:
    """
    Split comparisons into train and validation sets.

    Args:
        comparisons: List of comparison dicts
        train_ratio: Fraction of data for training

    Returns:
        Tuple of (train_comparisons, val_comparisons)
    """
    import random
    random.seed(42)  # For reproducibility

    # Shuffle comparisons
    shuffled = comparisons.copy()
    random.shuffle(shuffled)

    # Split
    split_idx = int(len(shuffled) * train_ratio)
    train_comparisons = shuffled[:split_idx]
    val_comparisons = shuffled[split_idx:]

    print(f"Train set: {len(train_comparisons)} comparisons")
    print(f"Val set: {len(val_comparisons)} comparisons")

    return train_comparisons, val_comparisons


def compute_statistics(comparisons: List[Dict]) -> Dict:
    """
    Compute statistics about the comparison dataset.

    Args:
        comparisons: List of comparison dicts

    Returns:
        Dict with statistics
    """
    from collections import Counter

    # Count labels
    label_counts = Counter(comp['label'] for comp in comparisons)

    # Count pair types
    pair_type_counts = Counter(comp.get('pair_type', 'unknown') for comp in comparisons)

    # Count category pairs
    category_pairs = Counter()
    for comp in comparisons:
        cat1 = comp.get('img1_category', '')
        cat2 = comp.get('img2_category', '')
        pair = f"{cat1}_vs_{cat2}"
        category_pairs[pair] += 1

    # Unique images
    images = set()
    for comp in comparisons:
        images.add(comp.get('img1', ''))
        images.add(comp.get('img2', ''))

    return {
        'total_samples': len(comparisons),
        'label_distribution': dict(label_counts),
        'pair_type_distribution': dict(pair_type_counts),
        'category_pair_distribution': dict(category_pairs),
        'unique_images': len(images),
    }


def save_training_data(
    train_comparisons: List[Dict],
    val_comparisons: List[Dict],
    all_comparisons: List[Dict],
    output_dir: str
):
    """
    Save prepared training data to JSON files.

    Args:
        train_comparisons: Training comparisons
        val_comparisons: Validation comparisons
        all_comparisons: All comparisons
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save train comparisons
    train_path = os.path.join(output_dir, 'train_comparisons.json')
    with open(train_path, 'w') as f:
        json.dump(train_comparisons, f, indent=2)
    print(f"Saved training comparisons to {train_path}")

    # Save val comparisons
    val_path = os.path.join(output_dir, 'val_comparisons.json')
    with open(val_path, 'w') as f:
        json.dump(val_comparisons, f, indent=2)
    print(f"Saved validation comparisons to {val_path}")

    # Save all comparisons
    all_path = os.path.join(output_dir, 'all_comparisons.json')
    with open(all_path, 'w') as f:
        json.dump(all_comparisons, f, indent=2)
    print(f"Saved all comparisons to {all_path}")

    # Compute and save statistics
    train_stats = compute_statistics(train_comparisons)
    val_stats = compute_statistics(val_comparisons)
    all_stats = compute_statistics(all_comparisons)

    metadata = {
        'created_at': datetime.now().isoformat(),
        'data_source': 'golden_ranking',
        'train_samples': len(train_comparisons),
        'val_samples': len(val_comparisons),
        'total_samples': len(all_comparisons),
        'train_statistics': train_stats,
        'val_statistics': val_stats,
        'all_statistics': all_stats,
    }

    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description='Prepare golden_ranking training data')
    parser.add_argument('--comparison-file', type=str,
                        default='/home/hitul/Desktop/quality-comparison-toolkit/data/gold_ranking_comparisons.json',
                        help='Path to gold_ranking_comparisons.json')
    parser.add_argument('--output-dir', type=str, default='training_data_golden_ranking',
                        help='Output directory for prepared training data')
    parser.add_argument('--train-split', type=float, default=0.8,
                        help='Fraction of data for training (default: 0.8)')

    args = parser.parse_args()

    print("=" * 80)
    print("Preparing Golden Ranking Training Data")
    print("=" * 80)
    print(f"Comparison file: {args.comparison_file}")
    print(f"Output directory: {args.output_dir}")
    print(f"Train split: {args.train_split}")
    print()

    # Load comparisons
    comparisons = load_golden_ranking_comparisons(args.comparison_file)

    # Add ground truth labels
    print("\nAdding ground truth labels based on quality rankings...")
    comparisons = add_ground_truth_labels(comparisons)

    # Split into train/val
    print("\nSplitting data into train and validation sets...")
    train_comparisons, val_comparisons = split_data(comparisons, args.train_split)

    # Save training data
    print("\nSaving prepared training data...")
    save_training_data(train_comparisons, val_comparisons, comparisons, args.output_dir)

    print("\n" + "=" * 80)
    print("Training data preparation complete!")
    print("=" * 80)
    print(f"Output directory: {args.output_dir}")
    print(f"Train samples: {len(train_comparisons)}")
    print(f"Val samples: {len(val_comparisons)}")
    print(f"Total samples: {len(comparisons)}")


if __name__ == '__main__':
    main()
