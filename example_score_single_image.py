"""
Example: Score a single image with calibrated 0-100 score.

Usage:
    python example_score_single_image.py path/to/image.png
"""

import sys
import torch
from model import load_model
from infer import score_single_image
from calibrate_scores import ScoreCalibrator


def score_image_calibrated(image_path: str) -> tuple:
    """
    Score a single image with both raw and calibrated (0-100) scores.

    Returns:
        (raw_score, calibrated_score)
    """
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(
        'results/best_model.pt',
        backbone_name='efficientnet_b0',
        device=device
    )

    # Load calibrator
    calibrator = ScoreCalibrator.load('score_calibrator.pkl')

    # Get raw score
    raw_score = score_single_image(model, image_path, input_size=448, device=device)

    # Calibrate to 0-100
    calibrated_score = calibrator.transform(raw_score)

    return raw_score, calibrated_score


def interpret_score(score: float) -> str:
    """Interpret quality score."""
    if score >= 75:
        return "Excellent Quality"
    elif score >= 50:
        return "Good Quality"
    elif score >= 25:
        return "Fair Quality"
    else:
        return "Poor Quality"


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python example_score_single_image.py path/to/image.png")
        sys.exit(1)

    image_path = sys.argv[1]

    print(f"\nScoring image: {image_path}")
    print("-" * 60)

    raw_score, calibrated_score = score_image_calibrated(image_path)

    print(f"Raw Score:        {raw_score:.4f}")
    print(f"Calibrated Score: {calibrated_score:.2f} / 100")
    print(f"Quality Rating:   {interpret_score(calibrated_score)}")
    print()
