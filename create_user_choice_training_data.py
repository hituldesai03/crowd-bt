"""
Create training data from raw comparison data using user choices as ground truth.
This ignores any predefined rules and uses actual user selections.
"""

import json
import random
from pathlib import Path
from typing import List, Dict

def load_raw_comparisons(raw_file: str) -> List[Dict]:
    """Load raw comparison data."""
    with open(raw_file, 'r') as f:
        data = json.load(f)
    return data['comparisons']


def convert_choice_to_label(choice: str) -> int:
    """
    Convert user choice to label for training.

    Args:
        choice: User's choice ('left', 'right', 'draw', 'tie', etc.)

    Returns:
        1 if left (img1 > img2), -1 if right (img2 > img1), 0 if draw/tie
    """
    choice = str(choice).lower().strip()

    if choice == 'left':
        return 1
    elif choice == 'right':
        return -1
    elif choice in ['tie', 'equal', 'draw']:
        return 0
    else:
        # Skip invalid/null choices
        return None


def process_comparisons(raw_comparisons: List[Dict]) -> List[Dict]:
    """
    Process raw comparisons into training format using user choices as GT.

    Args:
        raw_comparisons: List of raw comparison dicts with 'choice' field

    Returns:
        List of processed comparisons with 'label' and 'weight' fields
    """
    processed = []
    skipped = 0

    for idx, comp in enumerate(raw_comparisons):
        choice = comp.get('choice')

        # Skip if no valid choice
        if choice is None or choice == '':
            skipped += 1
            continue

        label = convert_choice_to_label(choice)

        # Skip invalid choices
        if label is None:
            skipped += 1
            continue

        # Create processed comparison
        processed_comp = {
            'img1': comp['img1'],
            'img2': comp['img2'],
            'img1_category': comp.get('img1_category', ''),
            'img2_category': comp.get('img2_category', ''),
            'type': comp.get('pair_type', 'unknown'),
            'pair_index': idx,
            'label': label,
            'user_choice': choice,  # Keep original choice for reference
            'weight': 1.0,  # Default weight
            'pair_type': comp.get('pair_type', 'unknown'),
        }

        # Optionally include metadata
        if 'timestamp' in comp:
            processed_comp['timestamp'] = comp['timestamp']
        if 'time_taken' in comp:
            processed_comp['time_taken'] = comp['time_taken']

        processed.append(processed_comp)

    print(f"Processed {len(processed)} comparisons, skipped {skipped} invalid entries")
    return processed


def split_train_val(comparisons: List[Dict], train_ratio: float = 0.8, seed: int = 42) -> tuple:
    """
    Split comparisons into train and validation sets, stratified by pair type.

    Args:
        comparisons: All comparisons
        train_ratio: Fraction for training (default: 0.8)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_comparisons, val_comparisons)
    """
    random.seed(seed)

    # Group by pair type for stratified split
    by_type = {}
    for comp in comparisons:
        pair_type = comp.get('pair_type', 'unknown')
        if pair_type not in by_type:
            by_type[pair_type] = []
        by_type[pair_type].append(comp)

    train_comps = []
    val_comps = []

    for pair_type, comps in by_type.items():
        random.shuffle(comps)
        split_idx = int(len(comps) * train_ratio)
        train_comps.extend(comps[:split_idx])
        val_comps.extend(comps[split_idx:])
        print(f"  {pair_type}: {split_idx} train, {len(comps) - split_idx} val")

    random.shuffle(train_comps)
    random.shuffle(val_comps)

    print(f"\nTotal split: {len(train_comps)} train, {len(val_comps)} val")
    return train_comps, val_comps


def save_comparisons(comparisons: List[Dict], output_file: str):
    """Save comparisons to JSON file."""
    with open(output_file, 'w') as f:
        json.dump(comparisons, f, indent=2)
    print(f"Saved {len(comparisons)} comparisons to {output_file}")


def create_metadata(all_comps: List[Dict]) -> Dict:
    """Create metadata about the dataset."""
    # Count by label
    label_counts = {1: 0, -1: 0, 0: 0}
    for comp in all_comps:
        label_counts[comp['label']] = label_counts.get(comp['label'], 0) + 1

    # Count by pair type
    type_counts = {}
    for comp in all_comps:
        pair_type = comp.get('pair_type', 'unknown')
        type_counts[pair_type] = type_counts.get(pair_type, 0) + 1

    metadata = {
        'total_comparisons': len(all_comps),
        'train_comparisons': len(all_comps),
        'val_comparisons': len(all_comps),
        'note': 'Both train and val contain the same data (no split)',
        'label_distribution': {
            'left_wins': label_counts[1],
            'right_wins': label_counts[-1],
            'draws': label_counts[0]
        },
        'pair_type_distribution': type_counts,
        'created_from': 'user_choices',
        'description': 'Training data created from actual user choices, not predefined rules'
    }

    return metadata


def main():
    """Main function to process raw comparisons and create training data."""

    # Paths
    raw_file = 'training_data_golden_ranking/raw/comparison_data.json'
    output_dir = Path('training_data_golden_ranking')

    print("Loading raw comparisons...")
    raw_comparisons = load_raw_comparisons(raw_file)
    print(f"Loaded {len(raw_comparisons)} raw comparisons")

    print("\nProcessing comparisons using user choices as ground truth...")
    processed_comparisons = process_comparisons(raw_comparisons)

    print("\nNOTE: Using ALL comparisons for both train and val (no split)")

    print("\nSaving processed data...")
    save_comparisons(processed_comparisons, output_dir / 'all_comparisons.json')
    save_comparisons(processed_comparisons, output_dir / 'train_comparisons.json')
    save_comparisons(processed_comparisons, output_dir / 'val_comparisons.json')

    print("\nCreating metadata...")
    metadata = create_metadata(processed_comparisons)
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {output_dir / 'metadata.json'}")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total comparisons: {metadata['total_comparisons']}")
    print(f"Train: {metadata['train_comparisons']}")
    print(f"Val: {metadata['val_comparisons']}")
    print(f"\nLabel distribution:")
    print(f"  Left wins (label=1): {metadata['label_distribution']['left_wins']}")
    print(f"  Right wins (label=-1): {metadata['label_distribution']['right_wins']}")
    print(f"  Draws (label=0): {metadata['label_distribution']['draws']}")
    print(f"\nPair type distribution:")
    for pair_type, count in metadata['pair_type_distribution'].items():
        print(f"  {pair_type}: {count}")
    print("="*60)
    print("\nTraining data created successfully!")
    print("BOTH train and val contain ALL comparisons (no split)")


if __name__ == '__main__':
    main()
