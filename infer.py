"""
Inference script for scoring single images with a trained model.
"""

import argparse
import os
import torch
from PIL import Image
from torchvision import transforms
from typing import List, Dict, Optional
from tqdm import tqdm
import json

from model import QualityScorer, load_model
from dataset import get_transforms


def score_single_image(
    model: QualityScorer,
    image_path: str,
    input_size: int = 448,
    device: str = "cuda"
) -> float:
    """
    Score a single image.

    Args:
        model: Trained QualityScorer model
        image_path: Path to image
        input_size: Model input size
        device: Device to run on

    Returns:
        Quality score (higher = better quality)
    """
    transform = get_transforms(input_size, is_train=False)

    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        score = model(image_tensor)

    return score.item()


def score_batch(
    model: QualityScorer,
    image_paths: List[str],
    input_size: int = 448,
    device: str = "cuda",
    batch_size: int = 32
) -> List[Dict]:
    """
    Score multiple images.

    Args:
        model: Trained QualityScorer model
        image_paths: List of image paths
        input_size: Model input size
        device: Device to run on
        batch_size: Batch size for inference

    Returns:
        List of dicts with 'path' and 'score'
    """
    transform = get_transforms(input_size, is_train=False)
    results = []

    # Process in batches
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Scoring"):
        batch_paths = image_paths[i:i + batch_size]
        batch_tensors = []

        for path in batch_paths:
            try:
                image = Image.open(path).convert('RGB')
                tensor = transform(image)
                batch_tensors.append(tensor)
            except Exception as e:
                print(f"Error loading {path}: {e}")
                results.append({'path': path, 'score': None, 'error': str(e)})
                continue

        if batch_tensors:
            batch = torch.stack(batch_tensors).to(device)

            with torch.no_grad():
                scores = model(batch)

            for path, score in zip(batch_paths[:len(batch_tensors)], scores):
                results.append({'path': path, 'score': score.item()})

    return results


def compare_images(
    model: QualityScorer,
    image1_path: str,
    image2_path: str,
    input_size: int = 448,
    device: str = "cuda"
) -> Dict:
    """
    Compare two images and determine which has higher quality.

    Returns:
        Dict with scores and comparison result
    """
    score1 = score_single_image(model, image1_path, input_size, device)
    score2 = score_single_image(model, image2_path, input_size, device)

    if score1 > score2 + 0.1:
        winner = "image1"
        confidence = "high" if score1 - score2 > 0.5 else "medium"
    elif score2 > score1 + 0.1:
        winner = "image2"
        confidence = "high" if score2 - score1 > 0.5 else "medium"
    else:
        winner = "draw"
        confidence = "low"

    return {
        'image1': image1_path,
        'image2': image2_path,
        'score1': score1,
        'score2': score2,
        'difference': score1 - score2,
        'winner': winner,
        'confidence': confidence
    }


def validate_gold_hierarchy(
    model: QualityScorer,
    gold_images: Dict[str, List[str]],
    input_size: int = 448,
    device: str = "cuda"
) -> Dict:
    """
    Validate that the model preserves the gold quality hierarchy.

    Expected ordering: goldensample > POS_5 > POS_4 > ... > POS_0

    Args:
        model: Trained model
        gold_images: Dict mapping category to list of image paths
        input_size: Model input size
        device: Device

    Returns:
        Validation results
    """
    # Score all gold images
    category_scores = {}

    for category, paths in gold_images.items():
        scores = []
        for path in paths:
            try:
                score = score_single_image(model, path, input_size, device)
                scores.append(score)
            except Exception as e:
                print(f"Error scoring {path}: {e}")

        if scores:
            category_scores[category] = {
                'mean': sum(scores) / len(scores),
                'min': min(scores),
                'max': max(scores),
                'count': len(scores)
            }

    # Expected order
    expected_order = ['goldensample', 'POS_5', 'POS_4', 'POS_3', 'POS_2', 'POS_1', 'POS_0']

    # Check if ordering is preserved
    violations = []
    for i in range(len(expected_order) - 1):
        cat1 = expected_order[i]
        cat2 = expected_order[i + 1]

        if cat1 in category_scores and cat2 in category_scores:
            if category_scores[cat1]['mean'] <= category_scores[cat2]['mean']:
                violations.append({
                    'higher': cat1,
                    'lower': cat2,
                    'higher_score': category_scores[cat1]['mean'],
                    'lower_score': category_scores[cat2]['mean']
                })

    return {
        'category_scores': category_scores,
        'violations': violations,
        'ordering_preserved': len(violations) == 0
    }


def main():
    parser = argparse.ArgumentParser(description='Score images with trained model')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--backbone', type=str, default='efficientnet_b4',
                        help='Backbone architecture')
    parser.add_argument('--input-size', type=int, default=448,
                        help='Input image size')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')

    # Input options
    parser.add_argument('--image', type=str, default=None,
                        help='Single image to score')
    parser.add_argument('--image-dir', type=str, default=None,
                        help='Directory of images to score')
    parser.add_argument('--compare', nargs=2, type=str, default=None,
                        metavar=('IMG1', 'IMG2'),
                        help='Compare two images')

    # Output
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for batch results')

    args = parser.parse_args()

    # Setup device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"

    # Load model
    print(f"Loading model from {args.checkpoint}")

    # Try to read backbone from checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'config' in checkpoint:
        backbone = checkpoint['config'].get('backbone_name', args.backbone)
        input_size = checkpoint['config'].get('input_size', args.input_size)
    else:
        backbone = args.backbone
        input_size = args.input_size

    model = load_model(args.checkpoint, backbone_name=backbone, device=device)
    print(f"Model loaded: {backbone}, input size: {input_size}")

    # Score single image
    if args.image:
        score = score_single_image(model, args.image, input_size, device)
        print(f"\nImage: {args.image}")
        print(f"Quality Score: {score:.4f}")

    # Compare two images
    elif args.compare:
        result = compare_images(model, args.compare[0], args.compare[1], input_size, device)
        print(f"\nComparison Results:")
        print(f"  Image 1: {result['image1']}")
        print(f"  Score 1: {result['score1']:.4f}")
        print(f"  Image 2: {result['image2']}")
        print(f"  Score 2: {result['score2']:.4f}")
        print(f"  Difference: {result['difference']:.4f}")
        print(f"  Winner: {result['winner']} ({result['confidence']} confidence)")

    # Score directory
    elif args.image_dir:
        # Find all images
        image_paths = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
            import glob
            image_paths.extend(glob.glob(os.path.join(args.image_dir, ext)))

        print(f"\nFound {len(image_paths)} images")

        results = score_batch(model, image_paths, input_size, device)

        # Sort by score
        results_sorted = sorted(
            [r for r in results if r['score'] is not None],
            key=lambda x: x['score'],
            reverse=True
        )

        print(f"\nTop 10 highest quality:")
        for r in results_sorted[:10]:
            print(f"  {os.path.basename(r['path'])}: {r['score']:.4f}")

        print(f"\nBottom 10 lowest quality:")
        for r in results_sorted[-10:]:
            print(f"  {os.path.basename(r['path'])}: {r['score']:.4f}")

        # Save results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results_sorted, f, indent=2)
            print(f"\nResults saved to {args.output}")

    else:
        print("Please specify --image, --compare, or --image-dir")


if __name__ == '__main__':
    main()
