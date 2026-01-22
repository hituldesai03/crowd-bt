"""
Data loader for Crowd-BT training.

Handles:
- Loading comparison data from S3 and local files
- Loading user reliability scores from S3
- Aggregating comparisons across multiple users
"""

import boto3
import json
import os
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from botocore.config import Config as BotoConfig

from config import default_config, get_quality_rank


def get_s3_client():
    """Get S3 client with timeout config."""
    config = BotoConfig(
        connect_timeout=10,
        read_timeout=30,
        retries={'max_attempts': 3}
    )
    return boto3.client('s3', config=config)


def list_users(bucket_name: str = None, data_prefix: str = None) -> List[str]:
    """
    List all users who have comparison data in S3.

    Returns:
        List of usernames
    """
    if bucket_name is None:
        bucket_name = default_config.data.bucket_name
    if data_prefix is None:
        data_prefix = default_config.data.data_prefix

    s3 = get_s3_client()

    try:
        # List all "directories" under the data prefix
        paginator = s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(
            Bucket=bucket_name,
            Prefix=data_prefix,
            Delimiter='/'
        )

        users = []
        for page in pages:
            for prefix in page.get('CommonPrefixes', []):
                # Extract username from prefix like "quack_v2_data_user_logs/username/"
                user_path = prefix['Prefix']
                username = user_path.replace(data_prefix, '').rstrip('/')
                if username and not username.endswith('.json'):
                    users.append(username)

        return users
    except Exception as e:
        print(f"Error listing users: {e}")
        return []


def load_user_comparisons(
    username: str,
    bucket_name: str = None,
    data_prefix: str = None
) -> Dict:
    """
    Load a user's comparison data from S3.

    Returns:
        Dict with comparison data including 'comparisons' list
    """
    if bucket_name is None:
        bucket_name = default_config.data.bucket_name
    if data_prefix is None:
        data_prefix = default_config.data.data_prefix

    s3 = get_s3_client()
    key = f"{data_prefix}{username}/comparison_data.json"

    try:
        obj = s3.get_object(Bucket=bucket_name, Key=key)
        return json.loads(obj['Body'].read().decode('utf-8'))
    except Exception as e:
        print(f"Error loading comparisons for {username}: {e}")
        return {'comparisons': []}


def compute_user_reliability(comparisons: List[Dict]) -> Tuple[float, int, int]:
    """
    Compute reliability score based on gold-gold pair performance.

    Returns:
        (reliability_score, correct_count, total_gold_pairs)
    """
    gold_comparisons = [c for c in comparisons if c.get('pair_type') == 'gold_gold']

    if not gold_comparisons:
        return 0.5, 0, 0  # Default reliability for users with no gold data

    correct = 0
    total = len(gold_comparisons)

    for comp in gold_comparisons:
        cat1 = comp.get('img1_category', '')
        cat2 = comp.get('img2_category', '')
        user_choice = comp.get('choice', '')

        rank1 = get_quality_rank(cat1)
        rank2 = get_quality_rank(cat2)

        # Determine expected choice
        if rank1 > rank2:
            expected = 'left'
        elif rank2 > rank1:
            expected = 'right'
        else:
            expected = 'draw'

        if user_choice == expected:
            correct += 1
        # Partial credit for close calls (within 1 level)
        elif user_choice == 'draw' and abs(rank1 - rank2) <= 1 and rank1 != 100 and rank2 != 100:
            correct += 0.5

    reliability = correct / total if total > 0 else 0.5
    return reliability, correct, total


def load_all_comparisons(
    bucket_name: str = None,
    data_prefix: str = None,
    min_reliability: float = 0.8,
    exclude_dummy: bool = True
) -> Tuple[List[Dict], Dict[str, float]]:
    """
    Load comparisons from all users with reliability scores.

    Args:
        bucket_name: S3 bucket name
        data_prefix: S3 prefix for user data
        min_reliability: Minimum reliability score to include user
        exclude_dummy: Whether to exclude dummy users

    Returns:
        Tuple of (all_comparisons, user_reliabilities)
        Each comparison dict includes 'annotator_id' and 'annotator_reliability'
    """
    users = list_users(bucket_name, data_prefix)

    if exclude_dummy:
        users = [u for u in users if not u.startswith('dummy')]

    all_comparisons = []
    user_reliabilities = {}

    for username in users:
        user_data = load_user_comparisons(username, bucket_name, data_prefix)
        comparisons = user_data.get('comparisons', [])

        if not comparisons:
            continue

        reliability, correct, total = compute_user_reliability(comparisons)
        user_reliabilities[username] = reliability

        if reliability < min_reliability:
            print(f"Skipping user {username} with reliability {reliability:.3f}")
            continue

        # Add annotator info to each comparison
        for comp in comparisons:
            comp['annotator_id'] = username
            comp['annotator_reliability'] = reliability
            all_comparisons.append(comp)

    print(f"Loaded {len(all_comparisons)} comparisons from {len(user_reliabilities)} users")
    return all_comparisons, user_reliabilities


def load_local_comparisons(
    comparison_files: List[str] = None,
    data_dir: str = None
) -> List[Dict]:
    """
    Load comparison pairs from local JSON files.

    Args:
        comparison_files: List of JSON files to load
        data_dir: Base data directory

    Returns:
        List of comparison dicts
    """
    if data_dir is None:
        data_dir = default_config.data.local_data_dir

    if comparison_files is None:
        comparison_files = [
            os.path.join(data_dir, 'gold_batch.json'),
            os.path.join(data_dir, 'Anchor_Anchor.json'),
            os.path.join(data_dir, 'Anchor_Regular.json'),
        ]

    all_comparisons = []

    for filepath in comparison_files:
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    comparisons = json.load(f)
                    all_comparisons.extend(comparisons)
                    print(f"Loaded {len(comparisons)} pairs from {filepath}")
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
        else:
            print(f"File not found: {filepath}")

    return all_comparisons


def aggregate_comparisons(comparisons: List[Dict]) -> Dict[Tuple[str, str], Dict]:
    """
    Aggregate multiple annotations for the same image pair.

    Returns:
        Dict mapping (img1, img2) -> aggregated data including:
        - 'left_votes': weighted votes for left
        - 'right_votes': weighted votes for right
        - 'draw_votes': weighted votes for draw
        - 'total_weight': sum of all weights
        - 'annotations': list of individual annotations
    """
    aggregated = defaultdict(lambda: {
        'left_votes': 0.0,
        'right_votes': 0.0,
        'draw_votes': 0.0,
        'total_weight': 0.0,
        'annotations': []
    })

    for comp in comparisons:
        img1 = comp.get('img1')
        img2 = comp.get('img2')
        choice = comp.get('choice')

        if not img1 or not img2 or not choice:
            continue

        # Ensure consistent ordering
        key = (img1, img2)

        # Weight by annotator reliability (default to 1.0)
        weight = comp.get('annotator_reliability', 1.0)

        agg = aggregated[key]
        agg['annotations'].append(comp)
        agg['total_weight'] += weight

        if choice == 'left':
            agg['left_votes'] += weight
        elif choice == 'right':
            agg['right_votes'] += weight
        elif choice == 'draw':
            agg['draw_votes'] += weight

        # Store metadata from first annotation
        if 'img1_category' not in agg:
            agg['img1'] = img1
            agg['img2'] = img2
            agg['pair_type'] = comp.get('pair_type', 'unknown')
            agg['img1_category'] = comp.get('img1_category', '')
            agg['img2_category'] = comp.get('img2_category', '')

    return dict(aggregated)


def get_comparison_label(aggregated: Dict, threshold: float = 0.6) -> Optional[int]:
    """
    Convert aggregated votes to a comparison label.

    Args:
        aggregated: Aggregated comparison data
        threshold: Minimum vote fraction for a clear winner

    Returns:
        1 if img1 > img2, -1 if img1 < img2, 0 if draw, None if unclear
    """
    total = aggregated['total_weight']
    if total == 0:
        return None

    left_frac = aggregated['left_votes'] / total
    right_frac = aggregated['right_votes'] / total
    draw_frac = aggregated['draw_votes'] / total

    if left_frac >= threshold:
        return 1  # img1 > img2
    elif right_frac >= threshold:
        return -1  # img1 < img2
    elif draw_frac >= threshold:
        return 0  # draw
    else:
        return None  # unclear - skip this pair


def prepare_training_data(
    comparisons: List[Dict],
    use_gold_labels: bool = True,
    vote_threshold: float = 0.6
) -> List[Dict]:
    """
    Prepare comparison data for training.

    Args:
        comparisons: List of comparison dicts
        use_gold_labels: Whether to use expected_choice for gold pairs
        vote_threshold: Threshold for aggregated vote labels

    Returns:
        List of training samples with 'img1', 'img2', 'label', 'weight'
    """
    training_data = []

    # Aggregate comparisons
    aggregated = aggregate_comparisons(comparisons)

    for key, agg in aggregated.items():
        img1, img2 = key

        # For gold-gold pairs, use expected choice if available
        if use_gold_labels and agg.get('pair_type') == 'gold_gold':
            # Get expected from any annotation
            for ann in agg['annotations']:
                expected = ann.get('expected_choice')
                if expected:
                    if expected == 'left':
                        label = 1
                    elif expected == 'right':
                        label = -1
                    else:
                        label = 0
                    break
            else:
                # Infer from categories
                cat1 = agg.get('img1_category', '')
                cat2 = agg.get('img2_category', '')
                rank1 = get_quality_rank(cat1)
                rank2 = get_quality_rank(cat2)

                if rank1 > rank2:
                    label = 1
                elif rank1 < rank2:
                    label = -1
                else:
                    label = 0

            weight = 1.0  # Gold labels are reliable
        else:
            # Use aggregated votes
            label = get_comparison_label(agg, vote_threshold)
            if label is None:
                continue  # Skip unclear comparisons

            # Weight by agreement level
            total = agg['total_weight']
            if label == 1:
                weight = agg['left_votes'] / total
            elif label == -1:
                weight = agg['right_votes'] / total
            else:
                weight = agg['draw_votes'] / total

        training_data.append({
            'img1': img1,
            'img2': img2,
            'label': label,
            'weight': weight,
            'pair_type': agg.get('pair_type', 'unknown'),
            'num_annotations': len(agg['annotations']),
        })

    print(f"Prepared {len(training_data)} training samples")
    return training_data
