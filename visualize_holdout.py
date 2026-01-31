"""
Inference and visualization script for image directories.

Loads images from a directory, runs model inference, and creates visualizations
with predicted quality scores overlaid on each image.

Supports GPU batching and CPU multiprocessing for faster processing.
"""

import argparse
import os
import glob
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from model import load_model
from infer import score_single_image
from dataset import get_transforms


def load_images_from_directory(image_dir: str) -> List[str]:
    """
    Load all images from a directory.

    Args:
        image_dir: Directory containing images

    Returns:
        List of full image paths
    """
    image_paths = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
        image_paths.extend(glob.glob(os.path.join(image_dir, ext)))

    # Sort for consistent ordering
    image_paths = sorted(image_paths)

    return image_paths


class ImageDataset(Dataset):
    """PyTorch Dataset for batched image scoring."""

    def __init__(self, image_paths: List[str], input_size: int = 448):
        self.image_paths = image_paths
        self.input_size = input_size
        self.transform = get_transforms(input_size, is_train=False)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        tensor = self.transform(image)
        return {'tensor': tensor, 'path': img_path, 'idx': idx}


def score_images_batched(
    model,
    image_paths: List[str],
    input_size: int = 448,
    device: str = "cuda",
    batch_size: int = 32,
    num_workers: int = 4
) -> List[Dict]:
    """
    Score images using GPU batching.

    Args:
        model: Trained model
        image_paths: List of image paths
        input_size: Model input size
        device: Device for inference
        batch_size: Batch size for GPU
        num_workers: Number of data loading workers

    Returns:
        List of dicts with 'path', 'score', 'idx'
    """
    dataset = ImageDataset(image_paths, input_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device == "cuda" else False
    )

    results = [None] * len(image_paths)

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Scoring images (batched)"):
            tensors = batch['tensor'].to(device)
            paths = batch['path']
            indices = batch['idx']

            scores = model(tensors).squeeze(-1)

            for i in range(len(scores)):
                idx = indices[i].item()
                results[idx] = {
                    'path': paths[i],
                    'score': scores[i].item(),
                    'idx': idx
                }

    return results


def create_annotated_image_worker(args):
    """Worker function for multiprocessing visualization creation."""
    image_path, score, output_path, font_size, score_format = args

    try:
        img = Image.open(image_path).convert('RGB')
        draw = ImageDraw.Draw(img)

        # Load font
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", font_size)
            except:
                font = ImageFont.load_default()

        score_text = score_format.format(score)

        try:
            bbox = draw.textbbox((0, 0), score_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except:
            text_width, text_height = 80, 40

        padding = 10
        x = padding
        y = padding

        # Background rectangle
        draw.rectangle(
            [x - 5, y - 5, x + text_width + 5, y + text_height + 5],
            fill=(0, 0, 0, 200)
        )

        # Color based on score
        normalized = (score + 2) / 4
        normalized = max(0, min(1, normalized))

        if normalized < 0.5:
            r = 255
            g = int(255 * (normalized * 2))
        else:
            r = int(255 * (1 - (normalized - 0.5) * 2))
            g = 255
        b = 0
        text_color = (r, g, b)

        draw.text((x, y), score_text, fill=text_color, font=font)
        img.save(output_path)

        return {'success': True, 'path': output_path}
    except Exception as e:
        return {'success': False, 'path': output_path, 'error': str(e)}


def create_annotated_image(
    image_path: str,
    score: float,
    output_path: str,
    font_size: int = 40,
    score_format: str = "{:.3f}"
):
    """
    Create image with quality score overlaid.

    Args:
        image_path: Input image path
        score: Quality score to display
        output_path: Where to save annotated image
        font_size: Size of score text
        score_format: Format string for score
    """
    # Load image
    img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img)

    # Try to load a nice font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", font_size)
        except:
            font = ImageFont.load_default()

    # Format score text
    score_text = score_format.format(score)

    # Get text size for background rectangle
    try:
        bbox = draw.textbbox((0, 0), score_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    except:
        # Fallback for older Pillow versions
        text_width, text_height = draw.textsize(score_text, font=font)

    # Position at top-left with padding
    padding = 10
    x = padding
    y = padding

    # Draw semi-transparent background rectangle
    bg_color = (0, 0, 0, 200)  # Black with transparency
    draw.rectangle(
        [x - 5, y - 5, x + text_width + 5, y + text_height + 5],
        fill=bg_color
    )

    # Determine text color based on score (green=high, red=low)
    # Normalize score to 0-1 range for coloring (assuming scores are roughly -2 to 2)
    normalized = (score + 2) / 4  # Map [-2, 2] -> [0, 1]
    normalized = max(0, min(1, normalized))  # Clamp

    # Color gradient from red to yellow to green
    if normalized < 0.5:
        r = 255
        g = int(255 * (normalized * 2))
    else:
        r = int(255 * (1 - (normalized - 0.5) * 2))
        g = 255
    b = 0
    text_color = (r, g, b)

    # Draw score text
    draw.text((x, y), score_text, fill=text_color, font=font)

    # Save
    img.save(output_path)


def score_and_visualize_holdout(
    model,
    holdout_images: List[str],
    output_dir: str,
    input_size: int = 448,
    device: str = "cuda"
) -> List[Dict]:
    """
    Score all holdout images and create visualizations (sequential).

    Returns:
        List of dicts with 'path', 'score', 'output_path'
    """
    os.makedirs(output_dir, exist_ok=True)

    results = []

    print(f"Scoring {len(holdout_images)} images...")
    for img_path in tqdm(holdout_images):
        try:
            # Score image
            score = score_single_image(model, img_path, input_size, device)

            # Create output filename
            basename = os.path.basename(img_path)
            output_path = os.path.join(output_dir, basename)

            # Create annotated image
            create_annotated_image(img_path, score, output_path)

            results.append({
                'path': img_path,
                'filename': basename,
                'score': score,
                'output_path': output_path
            })

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            results.append({
                'path': img_path,
                'filename': os.path.basename(img_path),
                'score': None,
                'error': str(e)
            })

    return results


def score_and_visualize_batched(
    model,
    image_paths: List[str],
    output_dir: str,
    input_size: int = 448,
    device: str = "cuda",
    batch_size: int = 32,
    num_workers: int = 4,
    viz_workers: int = None,
    font_size: int = 40,
    score_format: str = "{:.3f}"
) -> List[Dict]:
    """
    Score images with GPU batching and create visualizations with CPU multiprocessing.

    Args:
        model: Trained model
        image_paths: List of image paths
        output_dir: Output directory for visualizations
        input_size: Model input size
        device: Device for inference
        batch_size: Batch size for GPU inference
        num_workers: Workers for data loading
        viz_workers: Workers for visualization creation (default: CPU count)
        font_size: Font size for score overlay
        score_format: Format string for scores

    Returns:
        List of result dicts
    """
    os.makedirs(output_dir, exist_ok=True)

    if viz_workers is None:
        viz_workers = min(mp.cpu_count(), 8)

    # Step 1: Score all images with GPU batching
    print(f"Scoring {len(image_paths)} images with GPU batching...")
    print(f"  Batch size: {batch_size}, Data workers: {num_workers}")

    score_results = score_images_batched(
        model, image_paths, input_size, device, batch_size, num_workers
    )

    # Step 2: Create visualizations with CPU multiprocessing
    print(f"\nCreating visualizations with {viz_workers} CPU workers...")

    viz_tasks = []
    for result in score_results:
        if result is not None and result.get('score') is not None:
            basename = os.path.basename(result['path'])
            output_path = os.path.join(output_dir, basename)
            viz_tasks.append((
                result['path'],
                result['score'],
                output_path,
                font_size,
                score_format
            ))

    # Use multiprocessing for visualization creation
    viz_results = {}
    with ProcessPoolExecutor(max_workers=viz_workers) as executor:
        futures = {executor.submit(create_annotated_image_worker, task): task[0]
                   for task in viz_tasks}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Creating visualizations"):
            img_path = futures[future]
            try:
                viz_results[img_path] = future.result()
            except Exception as e:
                viz_results[img_path] = {'success': False, 'error': str(e)}

    # Combine results
    final_results = []
    for result in score_results:
        if result is None:
            continue

        img_path = result['path']
        basename = os.path.basename(img_path)
        output_path = os.path.join(output_dir, basename)

        final_result = {
            'path': img_path,
            'filename': basename,
            'score': result.get('score'),
            'output_path': output_path
        }

        viz_result = viz_results.get(img_path, {})
        if not viz_result.get('success', True):
            final_result['viz_error'] = viz_result.get('error', 'Unknown error')

        final_results.append(final_result)

    return final_results


def create_summary_visualization(
    results: List[Dict],
    output_path: str,
    max_images: int = 50,
    images_per_row: int = 10
):
    """
    Create a grid visualization of all images sorted by score.

    Args:
        results: List of result dicts
        output_path: Path to save summary figure
        max_images: Maximum number of images to show
        images_per_row: Number of images per row
    """
    # Filter valid results and sort by score
    valid_results = [r for r in results if r['score'] is not None]
    sorted_results = sorted(valid_results, key=lambda x: x['score'], reverse=True)

    # Limit to max_images
    display_results = sorted_results[:max_images]

    n_images = len(display_results)
    n_rows = (n_images + images_per_row - 1) // images_per_row

    fig, axes = plt.subplots(n_rows, images_per_row, figsize=(20, 2 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle('Images Ranked by Predicted Quality (High → Low)',
                 fontsize=16, fontweight='bold')

    for idx, result in enumerate(display_results):
        row = idx // images_per_row
        col = idx % images_per_row
        ax = axes[row, col]

        # Load and display image
        img = Image.open(result['path'])
        ax.imshow(img)
        ax.axis('off')

        # Add score as title
        score = result['score']
        color = 'green' if score > 0.5 else 'red' if score < -0.5 else 'orange'
        ax.set_title(f"{score:.3f}", fontsize=10, color=color, fontweight='bold')

    # Hide unused subplots
    for idx in range(n_images, n_rows * images_per_row):
        row = idx // images_per_row
        col = idx % images_per_row
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Summary visualization saved to {output_path}")


def create_score_distribution_plot(
    results: List[Dict],
    output_path: str
):
    """Create histogram of score distribution."""
    scores = [r['score'] for r in results if r['score'] is not None]

    plt.figure(figsize=(12, 6))

    # Histogram
    plt.subplot(1, 2, 1)
    plt.hist(scores, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    plt.xlabel('Quality Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Predicted Quality Scores', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)

    # Add statistics
    mean_score = np.mean(scores)
    median_score = np.median(scores)
    std_score = np.std(scores)
    plt.axvline(mean_score, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_score:.3f}')
    plt.axvline(median_score, color='green', linestyle='--', linewidth=2, label=f'Median: {median_score:.3f}')
    plt.legend()

    # Box plot
    plt.subplot(1, 2, 2)
    plt.boxplot(scores, vert=True)
    plt.ylabel('Quality Score', fontsize=12)
    plt.title('Score Distribution (Box Plot)', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)

    # Add statistics text
    stats_text = f"Mean: {mean_score:.3f}\nMedian: {median_score:.3f}\nStd: {std_score:.3f}\nMin: {min(scores):.3f}\nMax: {max(scores):.3f}"
    plt.text(1.15, np.median(scores), stats_text, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Score distribution plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize images with predicted quality scores'
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

    # Data args
    parser.add_argument('--image-dir', type=str, required=True,
                        help='Directory containing images to visualize')

    # Output args
    parser.add_argument('--output-dir', type=str, default='holdout_visualizations',
                        help='Directory to save annotated images')
    parser.add_argument('--summary-grid', type=str, default='holdout_summary.png',
                        help='Path for summary grid visualization')
    parser.add_argument('--distribution-plot', type=str, default='score_distribution.png',
                        help='Path for score distribution plot')
    parser.add_argument('--max-grid-images', type=int, default=100,
                        help='Maximum images to show in summary grid')

    # Font args
    parser.add_argument('--font-size', type=int, default=40,
                        help='Font size for score overlay')

    # Batching args
    parser.add_argument('--batched', action='store_true',
                        help='Use GPU batching and CPU multiprocessing (faster)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for GPU inference (default: 32)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Workers for data loading (default: 4)')
    parser.add_argument('--viz-workers', type=int, default=None,
                        help='Workers for visualization creation (default: CPU count)')

    args = parser.parse_args()

    # Setup device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"

    print("="*60)
    print("Image Quality Visualization")
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

    # Load images from directory
    print(f"\nLoading images from {args.image_dir}")
    holdout_images = load_images_from_directory(args.image_dir)
    print(f"Found {len(holdout_images)} images")

    # Score and create individual visualizations
    print(f"\nCreating annotated images in {args.output_dir}")

    if args.batched:
        print("Using GPU batching and CPU multiprocessing")
        results = score_and_visualize_batched(
            model,
            holdout_images,
            args.output_dir,
            input_size=input_size,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            viz_workers=args.viz_workers,
            font_size=args.font_size
        )
    else:
        results = score_and_visualize_holdout(
            model,
            holdout_images,
            args.output_dir,
            input_size=input_size,
            device=device
        )

    # Save results JSON
    results_json_path = os.path.join(args.output_dir, 'scores.json')
    with open(results_json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nScores saved to {results_json_path}")

    # Statistics
    valid_scores = [r['score'] for r in results if r['score'] is not None]
    if valid_scores:
        print("\n" + "="*60)
        print("Score Statistics")
        print("="*60)
        print(f"Total images: {len(results)}")
        print(f"Successfully scored: {len(valid_scores)}")
        print(f"Mean score: {np.mean(valid_scores):.4f}")
        print(f"Median score: {np.median(valid_scores):.4f}")
        print(f"Std score: {np.std(valid_scores):.4f}")
        print(f"Min score: {np.min(valid_scores):.4f}")
        print(f"Max score: {np.max(valid_scores):.4f}")

        # Top and bottom 5
        sorted_results = sorted(results, key=lambda x: x.get('score', float('-inf')), reverse=True)

        print(f"\nTop 5 highest quality:")
        for r in sorted_results[:5]:
            if r['score'] is not None:
                print(f"  {r['filename']}: {r['score']:.4f}")

        print(f"\nBottom 5 lowest quality:")
        for r in sorted_results[-5:]:
            if r['score'] is not None:
                print(f"  {r['filename']}: {r['score']:.4f}")

    # Create summary visualizations
    print("\n" + "="*60)
    print("Creating Summary Visualizations")
    print("="*60)

    summary_path = os.path.join(args.output_dir, args.summary_grid)
    create_summary_visualization(
        results,
        summary_path,
        max_images=args.max_grid_images
    )

    dist_path = os.path.join(args.output_dir, args.distribution_plot)
    create_score_distribution_plot(results, dist_path)

    print("\n" + "="*60)
    print("Visualization Complete!")
    print("="*60)
    print(f"Annotated images: {args.output_dir}")
    print(f"Summary grid: {summary_path}")
    print(f"Distribution plot: {dist_path}")
    print(f"Scores JSON: {results_json_path}")


if __name__ == '__main__':
    main()
