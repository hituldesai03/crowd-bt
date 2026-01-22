"""
Inference and visualization script for holdout set with CALIBRATED 0-100 scores.

Loads holdout images, runs model inference, calibrates scores to 0-100 range,
and creates visualizations with calibrated quality scores overlaid on each patch.
"""

import argparse
import os
import json
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from typing import List, Dict
import matplotlib.pyplot as plt

from model import load_model
from infer import score_single_image
from calibrate_scores import ScoreCalibrator


def load_holdout_images(holdout_file: str, image_dir: str) -> List[str]:
    """Load list of holdout images."""
    with open(holdout_file, 'r') as f:
        filenames = [line.strip() for line in f if line.strip()]

    image_paths = [os.path.join(image_dir, fname) for fname in filenames]
    existing_paths = [p for p in image_paths if os.path.exists(p)]

    if len(existing_paths) < len(image_paths):
        missing = len(image_paths) - len(existing_paths)
        print(f"Warning: {missing} images not found")

    return existing_paths


def create_annotated_image_calibrated(
    image_path: str,
    raw_score: float,
    calibrated_score: float,
    output_path: str,
    font_size: int = 40,
    show_raw: bool = False
):
    """
    Create image with calibrated quality score (0-100) overlaid.

    Args:
        image_path: Input image path
        raw_score: Raw model score
        calibrated_score: Calibrated score (0-100)
        output_path: Where to save annotated image
        font_size: Size of score text
        show_raw: Whether to also show raw score
    """
    img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img)

    # Load font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", int(font_size * 0.6))
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", font_size)
            font_small = ImageFont.load_default()
        except:
            font = ImageFont.load_default()
            font_small = font

    # Main score text (calibrated 0-100)
    score_text = f"{calibrated_score:.1f}"

    # Get text size
    try:
        bbox = draw.textbbox((0, 0), score_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    except:
        text_width, text_height = draw.textsize(score_text, font=font)

    # Position at top-left
    padding = 10
    x = padding
    y = padding

    # Background height (extra space if showing raw score)
    bg_height = text_height + 10
    if show_raw:
        bg_height += int(font_size * 0.6) + 5

    # Draw background rectangle
    draw.rectangle(
        [x - 5, y - 5, x + text_width + 5, y + bg_height],
        fill=(0, 0, 0, 200)
    )

    # Color based on calibrated score (0-100)
    if calibrated_score >= 75:
        text_color = (0, 255, 0)  # Green for high quality
    elif calibrated_score >= 50:
        text_color = (255, 255, 0)  # Yellow for medium
    elif calibrated_score >= 25:
        text_color = (255, 165, 0)  # Orange for low-medium
    else:
        text_color = (255, 0, 0)  # Red for low quality

    # Draw calibrated score
    draw.text((x, y), score_text, fill=text_color, font=font)

    # Optionally show raw score below
    if show_raw:
        raw_text = f"({raw_score:.2f})"
        draw.text((x, y + text_height + 5), raw_text, fill=(200, 200, 200), font=font_small)

    img.save(output_path)


def score_and_visualize_holdout_calibrated(
    model,
    calibrator: ScoreCalibrator,
    holdout_images: List[str],
    output_dir: str,
    input_size: int = 448,
    device: str = "cuda",
    show_raw: bool = False
) -> List[Dict]:
    """Score all holdout images with calibrated scores and create visualizations."""
    os.makedirs(output_dir, exist_ok=True)

    results = []

    print(f"Scoring {len(holdout_images)} holdout images...")
    for img_path in tqdm(holdout_images):
        try:
            # Score image (raw)
            raw_score = score_single_image(model, img_path, input_size, device)

            # Calibrate to 0-100
            calibrated_score = calibrator.transform(raw_score)

            # Create output filename
            basename = os.path.basename(img_path)
            output_path = os.path.join(output_dir, basename)

            # Create annotated image with calibrated score
            create_annotated_image_calibrated(
                img_path, raw_score, calibrated_score, output_path,
                show_raw=show_raw
            )

            results.append({
                'path': img_path,
                'filename': basename,
                'raw_score': raw_score,
                'calibrated_score': calibrated_score,
                'output_path': output_path
            })

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            results.append({
                'path': img_path,
                'filename': os.path.basename(img_path),
                'raw_score': None,
                'calibrated_score': None,
                'error': str(e)
            })

    return results


def create_summary_visualization_calibrated(
    results: List[Dict],
    output_path: str,
    max_images: int = 50,
    images_per_row: int = 10
):
    """Create a grid visualization sorted by calibrated score."""
    valid_results = [r for r in results if r['calibrated_score'] is not None]
    sorted_results = sorted(valid_results, key=lambda x: x['calibrated_score'], reverse=True)

    display_results = sorted_results[:max_images]

    n_images = len(display_results)
    n_rows = (n_images + images_per_row - 1) // images_per_row

    fig, axes = plt.subplots(n_rows, images_per_row, figsize=(20, 2 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle('Holdout Set: Patches Ranked by Quality Score (0-100 scale, High → Low)',
                 fontsize=16, fontweight='bold')

    for idx, result in enumerate(display_results):
        row = idx // images_per_row
        col = idx % images_per_row
        ax = axes[row, col]

        img = Image.open(result['path'])
        ax.imshow(img)
        ax.axis('off')

        score = result['calibrated_score']
        if score >= 75:
            color = 'green'
        elif score >= 50:
            color = 'orange'
        else:
            color = 'red'
        ax.set_title(f"{score:.1f}", fontsize=12, color=color, fontweight='bold')

    # Hide unused subplots
    for idx in range(n_images, n_rows * images_per_row):
        row = idx // images_per_row
        col = idx % images_per_row
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Summary visualization saved to {output_path}")


def create_score_distribution_plot_calibrated(
    results: List[Dict],
    output_path: str
):
    """Create histogram of calibrated score distribution."""
    scores = [r['calibrated_score'] for r in results if r['calibrated_score'] is not None]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax = axes[0]
    ax.hist(scores, bins=20, range=(0, 100), color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Quality Score (0-100)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Quality Scores', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.grid(axis='y', alpha=0.3)

    # Add quality zones
    ax.axvspan(0, 25, alpha=0.1, color='red', label='Poor')
    ax.axvspan(25, 50, alpha=0.1, color='orange', label='Fair')
    ax.axvspan(50, 75, alpha=0.1, color='yellow', label='Good')
    ax.axvspan(75, 100, alpha=0.1, color='green', label='Excellent')
    ax.legend(loc='upper left')

    mean_score = np.mean(scores)
    median_score = np.median(scores)
    ax.axvline(mean_score, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_score:.1f}')
    ax.axvline(median_score, color='blue', linestyle='--', linewidth=2, label=f'Median: {median_score:.1f}')

    # Box plot
    ax = axes[1]
    bp = ax.boxplot(scores, vert=True, widths=0.5, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][0].set_edgecolor('steelblue')
    bp['medians'][0].set_color('red')
    bp['medians'][0].set_linewidth(2)

    ax.set_ylabel('Quality Score (0-100)', fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_title('Score Distribution (Box Plot)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_xticklabels(['Quality Scores'])

    # Add statistics
    stats_text = (f"Mean: {mean_score:.1f}\n"
                  f"Median: {median_score:.1f}\n"
                  f"Std: {np.std(scores):.1f}\n"
                  f"Min: {min(scores):.1f}\n"
                  f"Max: {max(scores):.1f}")
    ax.text(1.15, 50, stats_text, fontsize=11, va='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Score distribution plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize holdout set with calibrated 0-100 quality scores'
    )

    # Model args
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--calibrator', type=str, required=True,
                        help='Path to fitted score calibrator (.pkl)')
    parser.add_argument('--backbone', type=str, default='efficientnet_b0',
                        help='Backbone architecture')
    parser.add_argument('--input-size', type=int, default=448,
                        help='Input image size')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')

    # Data args
    parser.add_argument('--holdout-file', type=str,
                        default='/home/hitul/Desktop/quality-comparison-toolkit/data/holdout.txt',
                        help='Path to holdout.txt')
    parser.add_argument('--image-dir', type=str,
                        default='/home/hitul/Desktop/quality-comparison-toolkit/data/iter_0',
                        help='Directory containing images')

    # Output args
    parser.add_argument('--output-dir', type=str, default='holdout_visualizations_calibrated',
                        help='Directory to save annotated images')
    parser.add_argument('--show-raw', action='store_true',
                        help='Show raw scores alongside calibrated scores')
    parser.add_argument('--max-grid-images', type=int, default=100,
                        help='Maximum images in summary grid')

    args = parser.parse_args()

    # Setup device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"

    print("="*60)
    print("Holdout Set Visualization (Calibrated 0-100)")
    print("="*60)

    # Load model
    print(f"\nLoading model from {args.checkpoint}")
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

    # Load calibrator
    print(f"\nLoading calibrator from {args.calibrator}")
    calibrator = ScoreCalibrator.load(args.calibrator)

    # Load holdout images
    print(f"\nLoading holdout images from {args.holdout_file}")
    holdout_images = load_holdout_images(args.holdout_file, args.image_dir)
    print(f"Found {len(holdout_images)} images")

    # Score and create visualizations
    print(f"\nCreating annotated images in {args.output_dir}")
    results = score_and_visualize_holdout_calibrated(
        model,
        calibrator,
        holdout_images,
        args.output_dir,
        input_size=input_size,
        device=device,
        show_raw=args.show_raw
    )

    # Save results
    results_json_path = os.path.join(args.output_dir, 'scores_calibrated.json')
    with open(results_json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nScores saved to {results_json_path}")

    # Statistics
    valid_results = [r for r in results if r['calibrated_score'] is not None]
    if valid_results:
        calibrated_scores = [r['calibrated_score'] for r in valid_results]

        print("\n" + "="*60)
        print("Calibrated Score Statistics (0-100)")
        print("="*60)
        print(f"Total images: {len(results)}")
        print(f"Successfully scored: {len(valid_results)}")
        print(f"Mean score: {np.mean(calibrated_scores):.2f}")
        print(f"Median score: {np.median(calibrated_scores):.2f}")
        print(f"Std score: {np.std(calibrated_scores):.2f}")
        print(f"Min score: {np.min(calibrated_scores):.2f}")
        print(f"Max score: {np.max(calibrated_scores):.2f}")

        # Count by quality tier
        excellent = sum(1 for s in calibrated_scores if s >= 75)
        good = sum(1 for s in calibrated_scores if 50 <= s < 75)
        fair = sum(1 for s in calibrated_scores if 25 <= s < 50)
        poor = sum(1 for s in calibrated_scores if s < 25)

        print(f"\nQuality Distribution:")
        print(f"  Excellent (75-100): {excellent} ({100*excellent/len(calibrated_scores):.1f}%)")
        print(f"  Good (50-75):       {good} ({100*good/len(calibrated_scores):.1f}%)")
        print(f"  Fair (25-50):       {fair} ({100*fair/len(calibrated_scores):.1f}%)")
        print(f"  Poor (0-25):        {poor} ({100*poor/len(calibrated_scores):.1f}%)")

        # Top and bottom 5
        sorted_results = sorted(valid_results, key=lambda x: x['calibrated_score'], reverse=True)

        print(f"\nTop 5 highest quality:")
        for r in sorted_results[:5]:
            print(f"  {r['filename']}: {r['calibrated_score']:.2f}")

        print(f"\nBottom 5 lowest quality:")
        for r in sorted_results[-5:]:
            print(f"  {r['filename']}: {r['calibrated_score']:.2f}")

    # Create visualizations
    print("\n" + "="*60)
    print("Creating Summary Visualizations")
    print("="*60)

    summary_path = os.path.join(args.output_dir, 'holdout_summary.png')
    create_summary_visualization_calibrated(results, summary_path, max_images=args.max_grid_images)

    dist_path = os.path.join(args.output_dir, 'score_distribution.png')
    create_score_distribution_plot_calibrated(results, dist_path)

    print("\n" + "="*60)
    print("Visualization Complete!")
    print("="*60)
    print(f"Annotated images: {args.output_dir}")
    print(f"Summary grid: {summary_path}")
    print(f"Distribution plot: {dist_path}")
    print(f"Scores JSON: {results_json_path}")


if __name__ == '__main__':
    main()
