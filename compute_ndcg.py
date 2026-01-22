"""
Compute NDCG@k and other ranking metrics between two score files.

This script compares two ranking files (e.g., ground truth vs predicted rankings)
and computes various ranking quality metrics including NDCG@k, Spearman's rho,
and Kendall's tau.
"""

import json
import argparse
import numpy as np
from typing import List, Dict, Tuple
from scipy.stats import spearmanr, kendalltau


def load_scores(filepath: str) -> List[Dict]:
    """Load scores from a JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def align_scores(ground_truth: List[Dict], predicted: List[Dict]) -> Tuple[List[float], List[float], List[str]]:
    """
    Align two score files by filename and extract scores.

    Args:
        ground_truth: List of score dicts (ground truth ranking)
        predicted: List of score dicts (predicted ranking)

    Returns:
        Tuple of (gt_scores, pred_scores, filenames) - aligned by filename
    """
    # Create dictionaries for fast lookup
    gt_dict = {item['filename']: item['calibrated_score'] for item in ground_truth}
    pred_dict = {item['filename']: item['calibrated_score'] for item in predicted}

    # Find common filenames
    common_filenames = set(gt_dict.keys()) & set(pred_dict.keys())

    if len(common_filenames) == 0:
        raise ValueError("No common filenames found between the two score files!")

    # Extract aligned scores
    filenames = sorted(common_filenames)
    gt_scores = [gt_dict[fn] for fn in filenames]
    pred_scores = [pred_dict[fn] for fn in filenames]

    print(f"Aligned {len(filenames)} common items")
    print(f"Ground truth file has {len(ground_truth)} items")
    print(f"Predicted file has {len(predicted)} items")

    if len(filenames) < len(ground_truth) or len(filenames) < len(predicted):
        missing_in_gt = set(pred_dict.keys()) - set(gt_dict.keys())
        missing_in_pred = set(gt_dict.keys()) - set(pred_dict.keys())
        if missing_in_gt:
            print(f"Warning: {len(missing_in_gt)} items in predicted file not found in ground truth")
        if missing_in_pred:
            print(f"Warning: {len(missing_in_pred)} items in ground truth file not found in predicted")

    return gt_scores, pred_scores, filenames


def compute_dcg(relevances: List[float], k: int = None) -> float:
    """
    Compute Discounted Cumulative Gain at k.

    Args:
        relevances: List of relevance scores (in ranked order)
        k: Cutoff position (None for all positions)

    Returns:
        DCG@k score
    """
    if k is None:
        k = len(relevances)
    else:
        k = min(k, len(relevances))

    dcg = 0.0
    for i in range(k):
        # DCG = sum(rel_i / log2(i+2))
        # Note: i+2 because i is 0-indexed and we want positions 1, 2, 3, ...
        dcg += relevances[i] / np.log2(i + 2)

    return dcg


def compute_ndcg_at_k(ground_truth_scores: List[float], predicted_scores: List[float], k: int = None) -> float:
    """
    Compute Normalized Discounted Cumulative Gain at k.

    Args:
        ground_truth_scores: Ground truth relevance scores
        predicted_scores: Predicted scores (used for ranking)
        k: Cutoff position (None for all positions)

    Returns:
        NDCG@k score (0 to 1, where 1 is perfect)
    """
    # Get indices that would sort predicted_scores in descending order
    pred_order = np.argsort(predicted_scores)[::-1]

    # Reorder ground truth scores by predicted ranking
    relevances = np.array(ground_truth_scores)[pred_order]

    # Compute DCG@k
    dcg = compute_dcg(relevances.tolist(), k)

    # Compute IDCG@k (ideal DCG - perfect ranking by ground truth)
    ideal_relevances = sorted(ground_truth_scores, reverse=True)
    idcg = compute_dcg(ideal_relevances, k)

    # Avoid division by zero
    if idcg == 0:
        return 0.0

    return dcg / idcg


def compute_ranking_metrics(
    ground_truth_scores: List[float],
    predicted_scores: List[float],
    k_values: List[int] = None
) -> Dict:
    """
    Compute comprehensive ranking metrics.

    Args:
        ground_truth_scores: Ground truth scores
        predicted_scores: Predicted scores
        k_values: List of k values for NDCG@k (default: [1, 3, 5, 10, 20, 50, 100])

    Returns:
        Dictionary of metric names to values
    """
    if k_values is None:
        k_values = [1, 3, 5, 10, 20, 50, 100]

    # Filter k_values to only include those <= number of items
    n_items = len(ground_truth_scores)
    k_values = [k for k in k_values if k <= n_items]

    metrics = {}

    # Compute NDCG@k for each k
    for k in k_values:
        ndcg_k = compute_ndcg_at_k(ground_truth_scores, predicted_scores, k)
        metrics[f'ndcg@{k}'] = ndcg_k

    # Compute NDCG for all items
    ndcg_all = compute_ndcg_at_k(ground_truth_scores, predicted_scores, None)
    metrics['ndcg@all'] = ndcg_all

    # Compute Spearman's rank correlation
    spearman_rho, spearman_p = spearmanr(ground_truth_scores, predicted_scores)
    metrics['spearman_rho'] = spearman_rho
    metrics['spearman_pvalue'] = spearman_p

    # Compute Kendall's tau
    kendall_tau, kendall_p = kendalltau(ground_truth_scores, predicted_scores)
    metrics['kendall_tau'] = kendall_tau
    metrics['kendall_pvalue'] = kendall_p

    # Compute Pearson correlation (on raw scores)
    pearson_corr = np.corrcoef(ground_truth_scores, predicted_scores)[0, 1]
    metrics['pearson_correlation'] = pearson_corr

    return metrics


def print_metrics_report(metrics: Dict, n_items: int):
    """Print a formatted report of the metrics."""
    print("\n" + "=" * 60)
    print("RANKING METRICS REPORT")
    print("=" * 60)
    print(f"Total items compared: {n_items}\n")

    print("NDCG@k (Normalized Discounted Cumulative Gain):")
    print("-" * 60)
    for key in sorted(metrics.keys()):
        if key.startswith('ndcg@'):
            print(f"  {key:15s}: {metrics[key]:.4f}")

    print("\nRank Correlation Metrics:")
    print("-" * 60)
    print(f"  Spearman's rho : {metrics['spearman_rho']:.4f} (p={metrics['spearman_pvalue']:.4e})")
    print(f"  Kendall's tau  : {metrics['kendall_tau']:.4f} (p={metrics['kendall_pvalue']:.4e})")
    print(f"  Pearson corr   : {metrics['pearson_correlation']:.4f}")

    print("\nInterpretation:")
    print("-" * 60)
    print("  NDCG@k: 1.0 = perfect ranking, 0.0 = worst possible")
    print("  Spearman/Kendall: 1.0 = perfect agreement, -1.0 = perfect disagreement")
    print("  NDCG@k focuses on top-k items, giving more weight to errors at top")
    print("=" * 60)


def save_metrics_json(metrics: Dict, output_path: str):
    """Save metrics to a JSON file."""
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Compute NDCG@k and ranking metrics between two score files'
    )

    parser.add_argument('ground_truth', type=str,
                        help='Path to ground truth scores JSON file')
    parser.add_argument('predicted', type=str,
                        help='Path to predicted scores JSON file')
    parser.add_argument('--k-values', type=int, nargs='+',
                        default=[1, 3, 5, 10, 20, 50, 100],
                        help='K values for NDCG@k (default: 1 3 5 10 20 50 100)')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save metrics JSON (optional)')
    parser.add_argument('--use-raw-scores', action='store_true',
                        help='Use raw_score instead of calibrated_score')

    args = parser.parse_args()

    print("Loading score files...")
    ground_truth_data = load_scores(args.ground_truth)
    predicted_data = load_scores(args.predicted)

    print("\nAligning scores by filename...")
    gt_scores, pred_scores, filenames = align_scores(ground_truth_data, predicted_data)

    print("\nComputing ranking metrics...")
    metrics = compute_ranking_metrics(gt_scores, pred_scores, args.k_values)

    # Add metadata
    metrics['n_items'] = len(filenames)
    metrics['ground_truth_file'] = args.ground_truth
    metrics['predicted_file'] = args.predicted

    # Print report
    print_metrics_report(metrics, len(filenames))

    # Save to JSON if requested
    if args.output:
        save_metrics_json(metrics, args.output)


if __name__ == '__main__':
    main()
