import json
import numpy as np
import argparse
from typing import List, Dict, Tuple
from scipy.stats import spearmanr, kendalltau, rankdata

def align_scores(ground_truth: List[Dict], predicted: List[Dict]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    gt_dict = {item['filename']: item['calibrated_score'] for item in ground_truth}
    pred_dict = {item['filename']: item['calibrated_score'] for item in predicted}
    common_filenames = sorted(set(gt_dict.keys()) & set(pred_dict.keys()))
    
    gt_scores = np.array([gt_dict[fn] for fn in common_filenames])
    pred_scores = np.array([pred_dict[fn] for fn in common_filenames])
    return gt_scores, pred_scores, common_filenames

def compute_linear_ndcg(gt_scores: np.ndarray, pred_scores: np.ndarray) -> float:
    """Computes Global NDCG using the Linear formula: sum(rel / log2(i+1))"""
    def get_dcg(scores, order_indices):
        ordered_relevance = scores[order_indices]
        ranks = np.arange(1, len(scores) + 1)
        # Linear DCG: numerator is just the relevance score
        return np.sum(ordered_relevance / np.log2(ranks + 1))

    # Sort indices by predicted scores (descending)
    pred_order = np.argsort(pred_scores)[::-1]
    # Sort indices by ground truth scores (descending) for Ideal DCG
    ideal_order = np.argsort(gt_scores)[::-1]

    actual_dcg = get_dcg(gt_scores, pred_order)
    ideal_dcg = get_dcg(gt_scores, ideal_order)
    
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0

def compute_global_metrics(gt_scores: np.ndarray, pred_scores: np.ndarray):
    n = len(gt_scores)
    
    # 1. Standard Correlations
    spearman, _ = spearmanr(gt_scores, pred_scores)
    kendall, _ = kendalltau(gt_scores, pred_scores)
    
    # 2. Global NDCG (Linear)
    ndcg_global = compute_linear_ndcg(gt_scores, pred_scores)
    
    # 3. R-squared (coefficient of determination) on scores
    ss_res = np.sum((gt_scores - pred_scores) ** 2)
    ss_tot = np.sum((gt_scores - np.mean(gt_scores)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # 4. Rank Displacement Metrics
    # rankdata handles ties; we subtract from n to get descending rank (highest score = rank 1)
    gt_ranks = n - rankdata(gt_scores, method='average') + 1
    pred_ranks = n - rankdata(pred_scores, method='average') + 1

    displacements = np.abs(gt_ranks - pred_ranks)
    mean_rank_error = np.mean(displacements)
    max_rank_error = np.max(displacements)

    return {
        "kendall_tau": kendall,
        "spearman_rho": spearman,
        "r_squared": r_squared,
        "ndcg_global": ndcg_global,
        "mean_rank_displacement": mean_rank_error,
        "max_rank_displacement": max_rank_error,
        "pairwise_accuracy": (kendall + 1) / 2 # Probability that a random pair is ordered correctly
    }

def main():
    parser = argparse.ArgumentParser(description='Global Ranking Alignment')
    parser.add_argument('ground_truth', type=str)
    parser.add_argument('predicted', type=str)
    args = parser.parse_args()

    # Load and align
    with open(args.ground_truth, 'r') as f: gt_data = json.load(f)
    with open(args.predicted, 'r') as f: pred_data = json.load(f)
    
    gt_s, pred_s, filenames = align_scores(gt_data, pred_data)
    results = compute_global_metrics(gt_s, pred_s)

    print("\n" + "="*50)
    print(f"GLOBAL RANKING ALIGNMENT (N={len(filenames)})")
    print("="*50)
    print(f"Kendall's Tau (Pairwise) : {results['kendall_tau']:.4f}")
    print(f"Spearman's Rho (Ranks)   : {results['spearman_rho']:.4f}")
    print(f"R-squared (Scores)       : {results['r_squared']:.4f}")
    print(f"Global NDCG (Linear)     : {results['ndcg_global']:.4f}")
    print(f"Pairwise Win Probability : {results['pairwise_accuracy']*100:.1f}%")
    print("-" * 50)
    print(f"Mean Rank Displacement   : {results['mean_rank_displacement']:.2f} spots")
    print(f"Max Rank Displacement    : {results['max_rank_displacement']:.2f} spots")
    print("="*50)
    print("Interpretation:")
    print(" - R-squared: Proportion of variance in ground truth explained by predictions.")
    print(" - Win Prob: Chance the model correctly picks the 'greater' of two items.")
    print(" - Displacement: Average distance between true rank and predicted rank.")

if __name__ == '__main__':
    main()  