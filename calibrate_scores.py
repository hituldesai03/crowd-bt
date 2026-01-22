"""
Score calibration utilities to map raw model scores to 0-100 range.

Two approaches:
1. Percentile-based: Use training set distribution (robust to outliers)
2. Min-Max: Simple linear scaling (sensitive to outliers)
"""

import argparse
import json
import numpy as np
import torch
from typing import List, Dict, Tuple
import pickle
import os
from tqdm import tqdm

from model import load_model
from infer import score_single_image


class ScoreCalibrator:
    """Calibrates raw model scores to 0-100 range."""

    def __init__(
        self,
        method: str = "percentile",
        lower_percentile: float = 5.0,
        upper_percentile: float = 95.0
    ):
        """
        Args:
            method: "percentile" or "minmax"
            lower_percentile: Percentile to map to 0 (default 5th)
            upper_percentile: Percentile to map to 100 (default 95th)
        """
        self.method = method
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile

        # Calibration parameters (fitted from training data)
        self.raw_min = None
        self.raw_max = None
        self.raw_lower = None  # Lower percentile value
        self.raw_upper = None  # Upper percentile value
        self.fitted = False

    def fit(self, raw_scores: List[float]):
        """
        Fit calibrator to observed score distribution.

        Args:
            raw_scores: List of raw model scores from training/validation set
        """
        raw_scores = np.array(raw_scores)

        self.raw_min = float(np.min(raw_scores))
        self.raw_max = float(np.max(raw_scores))
        self.raw_lower = float(np.percentile(raw_scores, self.lower_percentile))
        self.raw_upper = float(np.percentile(raw_scores, self.upper_percentile))

        self.fitted = True

        print(f"Calibrator fitted on {len(raw_scores)} scores:")
        print(f"  Raw min: {self.raw_min:.4f}")
        print(f"  Raw max: {self.raw_max:.4f}")
        print(f"  {self.lower_percentile}th percentile: {self.raw_lower:.4f}")
        print(f"  {self.upper_percentile}th percentile: {self.raw_upper:.4f}")

    def transform(self, raw_score: float) -> float:
        """
        Transform raw score to 0-100 range.

        Args:
            raw_score: Raw model output

        Returns:
            Calibrated score in [0, 100]
        """
        if not self.fitted:
            raise ValueError("Calibrator must be fitted before transform")

        if self.method == "percentile":
            # Map percentile range to 0-100
            if raw_score <= self.raw_lower:
                return 0.0
            elif raw_score >= self.raw_upper:
                return 100.0
            else:
                # Linear interpolation between percentiles
                ratio = (raw_score - self.raw_lower) / (self.raw_upper - self.raw_lower)
                return ratio * 100.0

        elif self.method == "minmax":
            # Simple min-max scaling
            if raw_score <= self.raw_min:
                return 0.0
            elif raw_score >= self.raw_max:
                return 100.0
            else:
                ratio = (raw_score - self.raw_min) / (self.raw_max - self.raw_min)
                return ratio * 100.0

        else:
            raise ValueError(f"Unknown method: {self.method}")

    def save(self, path: str):
        """Save calibrator to file."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Calibrator saved to {path}")

    @staticmethod
    def load(path: str) -> "ScoreCalibrator":
        """Load calibrator from file."""
        with open(path, 'rb') as f:
            calibrator = pickle.load(f)
        print(f"Calibrator loaded from {path}")
        return calibrator


def score_dataset_images(
    model,
    image_paths: List[str],
    input_size: int = 448,
    device: str = "cuda"
) -> List[float]:
    """Score all images and return raw scores."""
    scores = []

    for img_path in tqdm(image_paths, desc="Scoring images"):
        try:
            score = score_single_image(model, img_path, input_size, device)
            scores.append(score)
        except Exception as e:
            print(f"Error scoring {img_path}: {e}")

    return scores


def fit_calibrator_from_dataset(
    model,
    image_dir: str,
    method: str = "percentile",
    input_size: int = 448,
    device: str = "cuda",
    max_images: int = None
) -> ScoreCalibrator:
    """
    Fit calibrator from a dataset of images.

    Args:
        model: Trained model
        image_dir: Directory containing images
        method: Calibration method
        input_size: Model input size
        device: Device
        max_images: Maximum number of images to use (None = all)

    Returns:
        Fitted ScoreCalibrator
    """
    import glob

    # Find all images
    image_paths = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
        image_paths.extend(glob.glob(os.path.join(image_dir, ext)))

    if max_images is not None and len(image_paths) > max_images:
        import random
        random.seed(42)
        image_paths = random.sample(image_paths, max_images)

    print(f"\nScoring {len(image_paths)} images for calibration...")
    raw_scores = score_dataset_images(model, image_paths, input_size, device)

    # Fit calibrator
    calibrator = ScoreCalibrator(method=method)
    calibrator.fit(raw_scores)

    return calibrator


def main():
    parser = argparse.ArgumentParser(
        description='Calibrate model scores to 0-100 range'
    )

    # Model args
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--backbone', type=str, default='efficientnet_b0',
                        help='Backbone architecture')
    parser.add_argument('--input-size', type=int, default=448,
                        help='Input image size')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')

    # Calibration args
    parser.add_argument('--image-dir', type=str, required=True,
                        help='Directory of images to fit calibrator')
    parser.add_argument('--method', type=str, default='percentile',
                        choices=['percentile', 'minmax'],
                        help='Calibration method')
    parser.add_argument('--lower-percentile', type=float, default=5.0,
                        help='Lower percentile to map to 0')
    parser.add_argument('--upper-percentile', type=float, default=95.0,
                        help='Upper percentile to map to 100')
    parser.add_argument('--max-images', type=int, default=None,
                        help='Max images to use for calibration')

    # Output
    parser.add_argument('--output', type=str, default='score_calibrator.pkl',
                        help='Path to save calibrator')

    # Test calibration
    parser.add_argument('--test-scores', type=str, default=None,
                        help='Optional JSON file with scores to test calibration')

    args = parser.parse_args()

    # Setup device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"

    # Load model
    print(f"Loading model from {args.checkpoint}")
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if 'config' in checkpoint:
            backbone = checkpoint['config'].get('backbone_name', args.backbone)
            input_size = checkpoint['config'].get('input_size', args.input_size)
        else:
            backbone = args.backbone
            input_size = args.input_size
    except:
        backbone = args.backbone
        input_size = args.input_size

    model = load_model(args.checkpoint, backbone_name=backbone, device=device)
    print(f"Model loaded: {backbone}, input size: {input_size}")

    # Fit calibrator
    print("\n" + "="*60)
    print("Fitting Score Calibrator")
    print("="*60)

    calibrator = fit_calibrator_from_dataset(
        model,
        args.image_dir,
        method=args.method,
        input_size=input_size,
        device=device,
        max_images=args.max_images
    )

    # Save calibrator
    calibrator.save(args.output)

    # Test calibration if requested
    if args.test_scores:
        print("\n" + "="*60)
        print("Testing Calibration")
        print("="*60)

        with open(args.test_scores, 'r') as f:
            test_data = json.load(f)

        print(f"\nCalibrating {len(test_data)} scores...")

        for item in test_data[:10]:  # Show first 10
            raw_score = item['score']
            calibrated = calibrator.transform(raw_score)
            print(f"  {item['filename']:30s}: {raw_score:8.4f} → {calibrated:6.2f}")

        # Show distribution of calibrated scores
        calibrated_scores = [calibrator.transform(item['score']) for item in test_data]
        print(f"\nCalibrated score distribution:")
        print(f"  Mean: {np.mean(calibrated_scores):.2f}")
        print(f"  Median: {np.median(calibrated_scores):.2f}")
        print(f"  Std: {np.std(calibrated_scores):.2f}")
        print(f"  Min: {np.min(calibrated_scores):.2f}")
        print(f"  Max: {np.max(calibrated_scores):.2f}")


if __name__ == '__main__':
    main()
