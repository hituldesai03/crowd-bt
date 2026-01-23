"""
Prepare training data from S3 user comparisons.

This script:
1. Fetches all user comparison data from S3
2. Computes user reliability scores
3. Keeps each annotation as individual sample (no aggregation) - correct Crowd-BT approach
4. Each annotation includes user's reliability as fixed eta
5. Creates train/test split
6. Saves the prepared data locally
"""

import argparse
import json
import os
from datetime import datetime

from config import default_config
from data_loader import (
    load_all_comparisons,
    prepare_training_data,
    prepare_training_data_individual
)
from dataset import split_comparisons


def main():
    parser = argparse.ArgumentParser(description='Prepare S3 data for training')

    parser.add_argument('--bucket', type=str,
                        default='surface-quality-dataset',
                        help='S3 bucket name')
    parser.add_argument('--prefix', type=str,
                        default='quack_v2_data_user_logs/',
                        help='S3 data prefix')
    parser.add_argument('--min-reliability', type=float, default=0.3,
                        help='Minimum user reliability score to include user')
    parser.add_argument('--min-annotation-reliability', type=float, default=0.5,
                        help='Minimum reliability for individual annotations (used with --individual)')
    parser.add_argument('--vote-threshold', type=float, default=0.6,
                        help='Vote threshold for aggregated labels (used with --aggregate)')
    parser.add_argument('--train-split', type=float, default=0.8,
                        help='Train/validation split ratio')
    parser.add_argument('--exclude-dummy', action='store_true', default=True,
                        help='Exclude dummy users')
    parser.add_argument('--output-dir', type=str, default='training_data',
                        help='Output directory for prepared data')
    parser.add_argument('--aggregate', action='store_true',
                        help='Use aggregated mode (default is individual annotations)')

    args = parser.parse_args()
    use_individual = not args.aggregate

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*60)
    print("Fetching User Comparison Data from S3")
    print("="*60)
    print(f"Bucket: {args.bucket}")
    print(f"Prefix: {args.prefix}")
    print(f"Min User Reliability: {args.min_reliability}")
    print(f"Mode: {'Individual annotations (Crowd-BT)' if use_individual else 'Aggregated'}")
    if use_individual:
        print(f"Min Annotation Reliability: {args.min_annotation_reliability}")
    else:
        print(f"Vote Threshold: {args.vote_threshold}")
    print(f"Exclude Dummy: {args.exclude_dummy}")
    print()

    # Load all comparisons from S3
    print("Loading comparisons from all users...")
    all_comparisons, user_reliabilities = load_all_comparisons(
        bucket_name=args.bucket,
        data_prefix=args.prefix,
        min_reliability=args.min_reliability,
        exclude_dummy=args.exclude_dummy
    )

    print(f"\nLoaded {len(all_comparisons)} comparisons from {len(user_reliabilities)} users")

    # Print user reliability stats
    print("\nUser Reliability Scores:")
    print("-" * 60)
    sorted_users = sorted(user_reliabilities.items(), key=lambda x: x[1], reverse=True)
    for username, reliability in sorted_users:
        print(f"  {username:20s}: {reliability:.3f}")

    avg_reliability = sum(user_reliabilities.values()) / len(user_reliabilities)
    print(f"\nAverage Reliability: {avg_reliability:.3f}")

    # Save user reliabilities
    reliability_path = os.path.join(args.output_dir, 'user_reliabilities.json')
    with open(reliability_path, 'w') as f:
        json.dump(user_reliabilities, f, indent=2)
    print(f"Saved user reliabilities to {reliability_path}")

    # Prepare training data
    print("\n" + "="*60)
    print("Preparing Training Data")
    print("="*60)

    if use_individual:
        # Keep each annotation separate with user reliability as fixed eta
        # This is the correct Crowd-BT approach
        print("Using individual annotations (each annotation is a training sample)")
        print("Each annotation's eta = annotator's reliability score")
        training_data = prepare_training_data_individual(
            all_comparisons,
            min_reliability=args.min_annotation_reliability
        )
    else:
        # Aggregate votes across users (legacy mode)
        print("Using aggregated mode (votes combined into single label)")
        training_data = prepare_training_data(
            all_comparisons,
            use_gold_labels=True,
            vote_threshold=args.vote_threshold
        )

    # Print data statistics
    print("\nData Statistics:")
    print("-" * 60)

    pair_types = {}
    label_counts = {-1: 0, 0: 0, 1: 0}

    for sample in training_data:
        pair_type = sample.get('pair_type', 'unknown')
        pair_types[pair_type] = pair_types.get(pair_type, 0) + 1
        label = sample['label']
        label_counts[label] = label_counts.get(label, 0) + 1

    print(f"Total samples: {len(training_data)}")
    print(f"\nBy pair type:")
    for pair_type, count in sorted(pair_types.items()):
        print(f"  {pair_type:20s}: {count:5d} ({100*count/len(training_data):.1f}%)")

    print(f"\nBy label:")
    print(f"  img1 > img2 (label=1):  {label_counts[1]:5d} ({100*label_counts[1]/len(training_data):.1f}%)")
    print(f"  img1 < img2 (label=-1): {label_counts[-1]:5d} ({100*label_counts[-1]/len(training_data):.1f}%)")
    print(f"  Draw (label=0):         {label_counts[0]:5d} ({100*label_counts[0]/len(training_data):.1f}%)")

    # Save all training data
    all_data_path = os.path.join(args.output_dir, 'all_comparisons.json')
    with open(all_data_path, 'w') as f:
        json.dump(training_data, f, indent=2)
    print(f"\nSaved all training data to {all_data_path}")

    # Create train/test split
    print("\n" + "="*60)
    print("Creating Train/Validation Split")
    print("="*60)

    train_data, val_data = split_comparisons(
        training_data,
        train_ratio=args.train_split,
        stratify_by_pair_type=True,
        seed=42
    )

    # Save train and validation splits
    train_path = os.path.join(args.output_dir, 'train_comparisons.json')
    val_path = os.path.join(args.output_dir, 'val_comparisons.json')

    with open(train_path, 'w') as f:
        json.dump(train_data, f, indent=2)
    print(f"Saved training data to {train_path}")

    with open(val_path, 'w') as f:
        json.dump(val_data, f, indent=2)
    print(f"Saved validation data to {val_path}")

    # Save metadata
    metadata = {
        'created_at': datetime.now().isoformat(),
        'bucket_name': args.bucket,
        'data_prefix': args.prefix,
        'mode': 'individual' if use_individual else 'aggregated',
        'min_user_reliability': args.min_reliability,
        'min_annotation_reliability': args.min_annotation_reliability if use_individual else None,
        'vote_threshold': args.vote_threshold if not use_individual else None,
        'train_split': args.train_split,
        'exclude_dummy': args.exclude_dummy,
        'num_users': len(user_reliabilities),
        'avg_user_reliability': avg_reliability,
        'total_samples': len(training_data),
        'train_samples': len(train_data),
        'val_samples': len(val_data),
        'pair_type_distribution': pair_types,
        'label_distribution': label_counts,
        'uses_fixed_eta': use_individual,  # Each annotation's reliability is used as fixed eta
    }

    metadata_path = os.path.join(args.output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")

    print("\n" + "="*60)
    print("Data Preparation Complete!")
    print("="*60)
    print(f"Output directory: {args.output_dir}")
    print(f"Ready for training with {len(train_data)} train and {len(val_data)} val samples")
    print()


if __name__ == '__main__':
    main()
