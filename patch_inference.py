"""
Patch-Based CrowdBT Inference Pipeline

Runs CrowdBT inference on full images by:
1. Applying flat field correction (grayscale + FFC)
2. Extracting patches from images (e.g., 505x505 patches from 1516x1516 images)
3. Scoring each patch using the existing CrowdBT model
4. Aggregating scores using defect-weighted formula
5. Generating visualizations (heatmap, overlay, final score)
"""

import argparse
import json
import math
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm

from model import load_model
from dataset import get_transforms
from calibrate_scores import ScoreCalibrator


def compute_flat_field_corrected(
    image: np.ndarray,
    kernel_size: int = 101,
    method: str = 'mean',
    clip_factor: float = 0.9
) -> np.ndarray:
    """
    Flat-field correction for images - converts to grayscale and applies FFC.

    Args:
        image: Input image (BGR or RGB, uint8)
        kernel_size: Size of kernel for illumination estimation
        method: 'mean' (default), 'morphology', or 'gaussian'
        clip_factor: Factor for clipping highlights (0.9 default)

    Returns:
        Corrected grayscale image (uint8) stretched via NORM_MINMAX
    """
    # 1. Convert to float32 grayscale
    if len(image.shape) == 3:
        # Handle both BGR (cv2) and RGB (PIL) - convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    else:
        gray = image.astype(np.float32)

    # 2. Compute Illumination
    eps = 1e-6
    if method == 'mean':
        illumination = cv2.blur(gray, (kernel_size, kernel_size))
    elif method == "morphology":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        illumination = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    elif method == "gaussian":
        illumination = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    else:
        raise ValueError(f"Unknown method: {method}")

    # 3. Apply Correction
    corrected = gray / (illumination + eps)
    corrected *= np.mean(illumination)

    # Clip highlights to preserve bright areas
    if clip_factor < 1.0:
        max_allowed = clip_factor * corrected.max()
        corrected = np.clip(corrected, 0, max_allowed)

    # 4. Normalize for saving (STRETCH CONTRAST)
    corrected_normalized = cv2.normalize(
        corrected, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    return corrected_normalized


def preprocess_image_for_inference(
    image: Image.Image,
    kernel_size: int = 101,
    ffc_method: str = 'mean',
    clip_factor: float = 0.9
) -> Image.Image:
    """
    Preprocess image for CrowdBT inference: grayscale + flat field correction.

    The model was trained on grayscale FFC images, so we need to apply the same
    preprocessing before inference.

    Args:
        image: Input PIL Image (RGB)
        kernel_size: FFC kernel size
        ffc_method: FFC method ('mean', 'gaussian', 'morphology')
        clip_factor: FFC clip factor

    Returns:
        PIL Image (RGB, 3-channel from grayscale FFC)
    """
    # Convert PIL to numpy array
    img_array = np.array(image)

    # Apply flat field correction (returns grayscale uint8)
    ffc_gray = compute_flat_field_corrected(
        img_array,
        kernel_size=kernel_size,
        method=ffc_method,
        clip_factor=clip_factor
    )

    # Convert grayscale back to 3-channel RGB for the model
    # (model expects 3-channel input)
    ffc_rgb = cv2.cvtColor(ffc_gray, cv2.COLOR_GRAY2RGB)

    # Convert back to PIL Image
    return Image.fromarray(ffc_rgb)


@dataclass
class PatchInfo:
    """Information about a single patch extracted from an image."""
    patch_id: int
    row: int                    # 0-2 for 3x3 grid
    col: int                    # 0-2 for 3x3 grid
    x: int                      # Top-left x in original image
    y: int                      # Top-left y in original image
    raw_score: float = None
    calibrated_score: float = None


@dataclass
class PatchInferenceResult:
    """Result of patch-based inference on a full image."""
    image_path: str
    patches: List[PatchInfo]    # 9 patches for 3x3 grid
    lambda_param: float
    final_score: float          # Weighted aggregate (0-100)
    mean_patch_score: float     # Simple average
    min_patch_score: float
    max_patch_score: float
    std_patch_score: float = 0.0
    weights: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'image_path': self.image_path,
            'patches': [
                {
                    'patch_id': p.patch_id,
                    'row': p.row,
                    'col': p.col,
                    'x': p.x,
                    'y': p.y,
                    'raw_score': p.raw_score,
                    'calibrated_score': p.calibrated_score
                }
                for p in self.patches
            ],
            'lambda_param': self.lambda_param,
            'final_score': self.final_score,
            'mean_patch_score': self.mean_patch_score,
            'min_patch_score': self.min_patch_score,
            'max_patch_score': self.max_patch_score,
            'std_patch_score': self.std_patch_score,
            'weights': self.weights
        }


def extract_patches(
    image: Image.Image,
    patch_size: int = 505
) -> Tuple[List[Image.Image], List[PatchInfo]]:
    """
    Extract non-overlapping patches from image.

    For 1516x1516 image with 505x505 patches: 3x3 grid = 9 patches.
    Handle edge case: last patch may need adjustment (1516 vs 1515).

    Args:
        image: PIL Image to extract patches from
        patch_size: Size of each patch (default 505)

    Returns:
        Tuple of (list of patch images, list of PatchInfo objects)
    """
    width, height = image.size

    # Calculate number of patches in each dimension
    n_cols = width // patch_size
    n_rows = height // patch_size

    # Handle edge pixels that don't fit perfectly
    # For 1516 with patch_size 505: 1516 // 505 = 3, but 3*505 = 1515
    # We'll include partial overlap for the last patch if needed
    extra_x = width - (n_cols * patch_size)
    extra_y = height - (n_rows * patch_size)

    patches = []
    patch_infos = []
    patch_id = 0

    for row in range(n_rows):
        for col in range(n_cols):
            # Calculate top-left coordinates
            x = col * patch_size
            y = row * patch_size

            # For the last row/column, adjust to include edge pixels
            if col == n_cols - 1 and extra_x > 0:
                x = width - patch_size
            if row == n_rows - 1 and extra_y > 0:
                y = height - patch_size

            # Ensure we don't go out of bounds
            x = min(x, width - patch_size)
            y = min(y, height - patch_size)

            # Handle case where image is smaller than patch_size
            if x < 0 or y < 0:
                x = max(0, x)
                y = max(0, y)
                actual_width = min(patch_size, width - x)
                actual_height = min(patch_size, height - y)
                patch = image.crop((x, y, x + actual_width, y + actual_height))
                # Pad if needed
                if actual_width < patch_size or actual_height < patch_size:
                    padded = Image.new('RGB', (patch_size, patch_size), (0, 0, 0))
                    padded.paste(patch, (0, 0))
                    patch = padded
            else:
                patch = image.crop((x, y, x + patch_size, y + patch_size))

            patches.append(patch)
            patch_infos.append(PatchInfo(
                patch_id=patch_id,
                row=row,
                col=col,
                x=x,
                y=y
            ))
            patch_id += 1

    return patches, patch_infos


def score_patches(
    model,
    patches: List[Image.Image],
    patch_infos: List[PatchInfo],
    input_size: int = 448,
    device: str = "cuda",
    batch_size: int = 9
) -> List[PatchInfo]:
    """
    Run CrowdBT model inference on all patches.

    Args:
        model: Trained QualityScorer model
        patches: List of PIL Image patches
        patch_infos: List of PatchInfo objects to update
        input_size: Model input size for transforms
        device: Device to run inference on
        batch_size: Batch size for inference

    Returns:
        patch_infos with raw_score populated
    """
    transform = get_transforms(input_size, is_train=False)

    # Process patches in batches
    all_scores = []

    for i in range(0, len(patches), batch_size):
        batch_patches = patches[i:i + batch_size]
        batch_tensors = []

        for patch in batch_patches:
            tensor = transform(patch)
            batch_tensors.append(tensor)

        if batch_tensors:
            batch = torch.stack(batch_tensors).to(device)

            with torch.no_grad():
                scores = model(batch)

            for score in scores:
                all_scores.append(score.item())

    # Update patch_infos with scores
    for info, score in zip(patch_infos, all_scores):
        info.raw_score = score

    return patch_infos


def aggregate_scores_weighted(
    patch_infos: List[PatchInfo],
    lambda_param: float = 5.0
) -> Dict:
    """
    Aggregate patch scores using defect-weighted formula.

    For each patch i:
        s_i = calibrated_score / 100        # Normalize to 0-1
        d_i = 1 - s_i                        # Defect severity
        w_i = exp(λ * d_i)                   # Influence weight

    Final Score = (1 - Σ(w_i * d_i) / Σ(w_i)) * 100

    λ = 0  → simple average
    λ = 10 → severe defects dominate

    Args:
        patch_infos: List of PatchInfo objects with calibrated_score set
        lambda_param: Defect emphasis parameter (default 5.0)

    Returns:
        Dict with final_score, mean_score, min_score, max_score, weights
    """
    # Normalize scores to 0-1
    scores = [p.calibrated_score / 100.0 for p in patch_infos]

    # Defect severity (1 = worst, 0 = perfect)
    defects = [1.0 - s for s in scores]

    # Exponential weights (bad patches get higher weight)
    weights = [math.exp(lambda_param * d) for d in defects]

    # Weighted defect average
    weighted_defect_sum = sum(w * d for w, d in zip(weights, defects))
    total_weight = sum(weights)

    # Final score: 1 - weighted_average_defect, scaled to 0-100
    final_score = (1.0 - weighted_defect_sum / total_weight) * 100.0

    calibrated_scores = [p.calibrated_score for p in patch_infos]

    return {
        'final_score': final_score,
        'mean_score': sum(calibrated_scores) / len(calibrated_scores),
        'min_score': min(calibrated_scores),
        'max_score': max(calibrated_scores),
        'std_score': float(np.std(calibrated_scores)),
        'weights': weights
    }


def get_quality_color(score: float) -> Tuple[int, int, int]:
    """
    Get RGB color based on quality score (0-100).

    Returns:
        RGB tuple (0-255 for each channel)
    """
    if score >= 75:
        return (0, 200, 0)      # Green - Excellent
    elif score >= 50:
        return (255, 200, 0)    # Yellow - Good
    elif score >= 25:
        return (255, 128, 0)    # Orange - Fair
    else:
        return (255, 0, 0)      # Red - Poor


def get_quality_tier(score: float) -> str:
    """Get quality tier label based on score."""
    if score >= 75:
        return "Excellent"
    elif score >= 50:
        return "Good"
    elif score >= 25:
        return "Fair"
    else:
        return "Poor"


def create_patch_heatmap(
    result: PatchInferenceResult,
    output_path: str
):
    """
    Create a 3x3 grid heatmap colored by score (RdYlGn colormap).

    Args:
        result: PatchInferenceResult with patch scores
        output_path: Path to save the heatmap image
    """
    # Determine grid size from patches
    max_row = max(p.row for p in result.patches)
    max_col = max(p.col for p in result.patches)
    n_rows = max_row + 1
    n_cols = max_col + 1

    # Create score matrix
    score_matrix = np.zeros((n_rows, n_cols))
    for patch in result.patches:
        score_matrix[patch.row, patch.col] = patch.calibrated_score

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))

    # Use RdYlGn colormap (Red-Yellow-Green)
    cmap = plt.cm.RdYlGn
    norm = mcolors.Normalize(vmin=0, vmax=100)

    # Create heatmap
    im = ax.imshow(score_matrix, cmap=cmap, norm=norm, aspect='equal')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Quality Score (0-100)', fontsize=12)

    # Annotate each cell with score value
    for patch in result.patches:
        score = patch.calibrated_score
        # Use white text for dark colors, black for light
        text_color = 'white' if score < 40 else 'black'
        ax.text(
            patch.col, patch.row, f'{score:.1f}',
            ha='center', va='center',
            fontsize=14, fontweight='bold',
            color=text_color
        )

    # Set labels
    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_rows))
    ax.set_xticklabels([f'Col {i}' for i in range(n_cols)])
    ax.set_yticklabels([f'Row {i}' for i in range(n_rows)])

    # Title with final score
    tier = get_quality_tier(result.final_score)
    ax.set_title(
        f'Patch Quality Heatmap\nFinal Score: {result.final_score:.1f} ({tier})',
        fontsize=14, fontweight='bold'
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_stitched_overlay(
    image_path: str,
    result: PatchInferenceResult,
    output_path: str,
    alpha: float = 0.3
):
    """
    Create original image with semi-transparent color overlay per patch.

    Args:
        image_path: Path to original image
        result: PatchInferenceResult with patch scores
        output_path: Path to save overlay image
        alpha: Transparency level (0-1) for overlay
    """
    # Load original image
    img = Image.open(image_path).convert('RGBA')
    width, height = img.size

    # Create overlay layer
    overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)

    # Determine patch size from result
    if len(result.patches) > 1:
        # Infer patch size from coordinates
        patches_sorted = sorted(result.patches, key=lambda p: (p.row, p.col))
        if len(patches_sorted) > 1:
            first_patch = patches_sorted[0]
            second_patch = patches_sorted[1]
            if second_patch.col > first_patch.col:
                patch_width = second_patch.x - first_patch.x
            else:
                patch_width = patches_sorted[result.patches[0].col + 1].x - first_patch.x if len(patches_sorted) > 1 else 505

            # For rows
            for p in patches_sorted:
                if p.row > first_patch.row:
                    patch_height = p.y - first_patch.y
                    break
            else:
                patch_height = patch_width
        else:
            patch_width = patch_height = 505
    else:
        patch_width = patch_height = 505

    # Draw colored rectangles for each patch
    for patch in result.patches:
        color = get_quality_color(patch.calibrated_score)
        # Calculate alpha value (0-255)
        alpha_int = int(alpha * 255)

        x1 = patch.x
        y1 = patch.y
        x2 = min(patch.x + patch_width, width)
        y2 = min(patch.y + patch_height, height)

        draw_overlay.rectangle(
            [x1, y1, x2, y2],
            fill=(*color, alpha_int)
        )

    # Composite overlay onto original
    img_with_overlay = Image.alpha_composite(img, overlay)

    # Convert back to RGB for drawing text
    img_rgb = img_with_overlay.convert('RGB')
    draw = ImageDraw.Draw(img_rgb)

    # Load font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 24)
            font_large = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 36)
        except:
            font = ImageFont.load_default()
            font_large = font

    # Draw grid lines
    for patch in result.patches:
        x1 = patch.x
        y1 = patch.y
        x2 = min(patch.x + patch_width, width)
        y2 = min(patch.y + patch_height, height)

        # Draw rectangle outline
        draw.rectangle([x1, y1, x2, y2], outline=(255, 255, 255), width=2)

        # Draw score text in center of patch
        score_text = f'{patch.calibrated_score:.1f}'
        try:
            bbox = draw.textbbox((0, 0), score_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except:
            text_width, text_height = 50, 20

        text_x = x1 + (x2 - x1 - text_width) // 2
        text_y = y1 + (y2 - y1 - text_height) // 2

        # Draw text with background
        padding = 5
        draw.rectangle(
            [text_x - padding, text_y - padding, text_x + text_width + padding, text_y + text_height + padding],
            fill=(0, 0, 0, 200)
        )
        draw.text((text_x, text_y), score_text, fill=(255, 255, 255), font=font)

    # Draw final score banner at top
    tier = get_quality_tier(result.final_score)
    banner_text = f'Final Score: {result.final_score:.1f} ({tier})'
    try:
        bbox = draw.textbbox((0, 0), banner_text, font=font_large)
        banner_width = bbox[2] - bbox[0]
        banner_height = bbox[3] - bbox[1]
    except:
        banner_width, banner_height = 300, 40

    banner_x = (width - banner_width) // 2
    banner_y = 10

    # Banner background
    color = get_quality_color(result.final_score)
    draw.rectangle(
        [banner_x - 15, banner_y - 5, banner_x + banner_width + 15, banner_y + banner_height + 10],
        fill=(*color, 230)
    )
    draw.text((banner_x, banner_y), banner_text, fill=(255, 255, 255), font=font_large)

    img_rgb.save(output_path)


def create_score_summary(
    result: PatchInferenceResult,
    output_path: str
):
    """
    Create summary visualization with final score, statistics, and histogram.

    Args:
        result: PatchInferenceResult with all patch scores
        output_path: Path to save summary image
    """
    fig = plt.figure(figsize=(12, 8))

    # Create grid layout
    gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1], width_ratios=[1, 1])

    # 1. Large final score display (top left)
    ax_score = fig.add_subplot(gs[0, 0])
    ax_score.axis('off')

    tier = get_quality_tier(result.final_score)
    color = {
        'Excellent': 'green',
        'Good': 'gold',
        'Fair': 'orange',
        'Poor': 'red'
    }.get(tier, 'gray')

    # Draw large score
    ax_score.text(
        0.5, 0.6, f'{result.final_score:.1f}',
        ha='center', va='center',
        fontsize=72, fontweight='bold',
        color=color, transform=ax_score.transAxes
    )
    ax_score.text(
        0.5, 0.25, tier,
        ha='center', va='center',
        fontsize=36, fontweight='bold',
        color=color, transform=ax_score.transAxes
    )
    ax_score.text(
        0.5, 0.05, f'λ = {result.lambda_param}',
        ha='center', va='center',
        fontsize=14, color='gray', transform=ax_score.transAxes
    )
    ax_score.set_title('Final Quality Score', fontsize=16, fontweight='bold', pad=20)

    # 2. Statistics (top right)
    ax_stats = fig.add_subplot(gs[0, 1])
    ax_stats.axis('off')

    stats_text = (
        f"Statistics\n"
        f"─────────────────\n"
        f"Mean Score:    {result.mean_patch_score:.1f}\n"
        f"Min Score:     {result.min_patch_score:.1f}\n"
        f"Max Score:     {result.max_patch_score:.1f}\n"
        f"Std Dev:       {result.std_patch_score:.1f}\n"
        f"─────────────────\n"
        f"Patches:       {len(result.patches)}\n"
        f"Grid Size:     {max(p.row for p in result.patches) + 1}×{max(p.col for p in result.patches) + 1}\n"
    )

    ax_stats.text(
        0.1, 0.5, stats_text,
        ha='left', va='center',
        fontsize=14, family='monospace',
        transform=ax_stats.transAxes
    )
    ax_stats.set_title('Patch Statistics', fontsize=16, fontweight='bold', pad=20)

    # 3. Histogram of patch scores (bottom left)
    ax_hist = fig.add_subplot(gs[1, 0])
    scores = [p.calibrated_score for p in result.patches]

    # Color bars by quality tier
    _, bins, patches_hist = ax_hist.hist(scores, bins=10, range=(0, 100), edgecolor='black', alpha=0.7)
    for patch_h, left, right in zip(patches_hist, bins[:-1], bins[1:]):
        mid = (left + right) / 2
        if mid >= 75:
            patch_h.set_facecolor('green')
        elif mid >= 50:
            patch_h.set_facecolor('gold')
        elif mid >= 25:
            patch_h.set_facecolor('orange')
        else:
            patch_h.set_facecolor('red')

    ax_hist.set_xlabel('Quality Score', fontsize=12)
    ax_hist.set_ylabel('Count', fontsize=12)
    ax_hist.set_title('Patch Score Distribution', fontsize=14, fontweight='bold')
    ax_hist.set_xlim(0, 100)
    ax_hist.grid(axis='y', alpha=0.3)

    # 4. Patch score bar chart (bottom right)
    ax_bars = fig.add_subplot(gs[1, 1])
    patch_labels = [f'R{p.row}C{p.col}' for p in result.patches]
    bar_colors = [
        'green' if p.calibrated_score >= 75 else
        'gold' if p.calibrated_score >= 50 else
        'orange' if p.calibrated_score >= 25 else
        'red' for p in result.patches
    ]

    ax_bars.bar(patch_labels, scores, color=bar_colors, edgecolor='black', alpha=0.7)
    ax_bars.set_xlabel('Patch Position', fontsize=12)
    ax_bars.set_ylabel('Score', fontsize=12)
    ax_bars.set_title('Individual Patch Scores', fontsize=14, fontweight='bold')
    ax_bars.set_ylim(0, 100)
    ax_bars.axhline(y=result.final_score, color='blue', linestyle='--', linewidth=2, label='Final Score')
    ax_bars.axhline(y=result.mean_patch_score, color='red', linestyle=':', linewidth=2, label='Mean')
    ax_bars.legend(loc='upper right', fontsize=10)
    ax_bars.tick_params(axis='x', rotation=45)
    ax_bars.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def process_single_image(
    model,
    calibrator: ScoreCalibrator,
    image_path: str,
    patch_size: int = 505,
    lambda_param: float = 5.0,
    input_size: int = 448,
    device: str = "cuda",
    output_dir: str = None,
    save_visualizations: bool = True,
    ffc_kernel_size: int = 101,
    ffc_method: str = 'mean',
    ffc_clip_factor: float = 0.9
) -> PatchInferenceResult:
    """
    Process a single image through the patch-based inference pipeline.

    Args:
        model: Trained QualityScorer model
        calibrator: Fitted ScoreCalibrator
        image_path: Path to input image
        patch_size: Size of patches to extract
        lambda_param: Defect emphasis parameter
        input_size: Model input size
        device: Device to run inference on
        output_dir: Directory to save outputs
        save_visualizations: Whether to save visualization files
        ffc_kernel_size: Kernel size for flat field correction
        ffc_method: FFC method ('mean', 'gaussian', 'morphology')
        ffc_clip_factor: FFC highlight clipping factor

    Returns:
        PatchInferenceResult with all scores and aggregated results
    """
    # Load image
    image = Image.open(image_path).convert('RGB')

    # Apply flat field correction (grayscale + FFC -> 3-channel RGB)
    image = preprocess_image_for_inference(
        image,
        kernel_size=ffc_kernel_size,
        ffc_method=ffc_method,
        clip_factor=ffc_clip_factor
    )

    # Extract patches
    patches, patch_infos = extract_patches(image, patch_size)

    # Score patches
    patch_infos = score_patches(model, patches, patch_infos, input_size, device)

    # Calibrate scores
    for patch_info in patch_infos:
        patch_info.calibrated_score = calibrator.transform(patch_info.raw_score)

    # Aggregate scores
    agg_result = aggregate_scores_weighted(patch_infos, lambda_param)

    # Create result object
    result = PatchInferenceResult(
        image_path=image_path,
        patches=patch_infos,
        lambda_param=lambda_param,
        final_score=agg_result['final_score'],
        mean_patch_score=agg_result['mean_score'],
        min_patch_score=agg_result['min_score'],
        max_patch_score=agg_result['max_score'],
        std_patch_score=agg_result['std_score'],
        weights=agg_result['weights']
    )

    # Save visualizations if requested
    if save_visualizations and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        basename = os.path.splitext(os.path.basename(image_path))[0]

        # Heatmap
        heatmap_path = os.path.join(output_dir, f'{basename}_heatmap.png')
        create_patch_heatmap(result, heatmap_path)

        # Overlay
        overlay_path = os.path.join(output_dir, f'{basename}_overlay.png')
        create_stitched_overlay(image_path, result, overlay_path)

        # Summary
        summary_path = os.path.join(output_dir, f'{basename}_summary.png')
        create_score_summary(result, summary_path)

        # JSON result
        json_path = os.path.join(output_dir, f'{basename}_result.json')
        with open(json_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)

    return result


def process_image_directory(
    model,
    calibrator: ScoreCalibrator,
    image_dir: str,
    patch_size: int = 505,
    lambda_param: float = 5.0,
    input_size: int = 448,
    device: str = "cuda",
    output_dir: str = None,
    save_visualizations: bool = True,
    ffc_kernel_size: int = 101,
    ffc_method: str = 'mean',
    ffc_clip_factor: float = 0.9
) -> List[PatchInferenceResult]:
    """
    Process all images in a directory through the patch-based inference pipeline.

    Args:
        model: Trained QualityScorer model
        calibrator: Fitted ScoreCalibrator
        image_dir: Directory containing images
        patch_size: Size of patches to extract
        lambda_param: Defect emphasis parameter
        input_size: Model input size
        device: Device to run inference on
        output_dir: Directory to save outputs
        save_visualizations: Whether to save visualization files
        ffc_kernel_size: Kernel size for flat field correction
        ffc_method: FFC method ('mean', 'gaussian', 'morphology')
        ffc_clip_factor: FFC highlight clipping factor

    Returns:
        List of PatchInferenceResult for all processed images
    """
    import glob

    # Find all images
    image_paths = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
        image_paths.extend(glob.glob(os.path.join(image_dir, ext)))

    if not image_paths:
        print(f"No images found in {image_dir}")
        return []

    print(f"Found {len(image_paths)} images to process")

    results = []
    for image_path in tqdm(image_paths, desc="Processing images"):
        try:
            result = process_single_image(
                model, calibrator, image_path,
                patch_size=patch_size,
                lambda_param=lambda_param,
                input_size=input_size,
                device=device,
                output_dir=output_dir,
                save_visualizations=save_visualizations,
                ffc_kernel_size=ffc_kernel_size,
                ffc_method=ffc_method,
                ffc_clip_factor=ffc_clip_factor
            )
            results.append(result)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    # Save combined results
    if output_dir and results:
        combined_path = os.path.join(output_dir, 'all_results.json')
        with open(combined_path, 'w') as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
        print(f"Combined results saved to {combined_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Patch-based CrowdBT inference pipeline for full images'
    )

    # Input arguments
    parser.add_argument('--image', type=str, default=None,
                        help='Single image to process')
    parser.add_argument('--image-dir', type=str, default=None,
                        help='Directory of images to process')

    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--calibrator', type=str, required=True,
                        help='Path to fitted score calibrator (.pkl)')
    parser.add_argument('--backbone', type=str, default='efficientnet_b0',
                        help='Backbone architecture')
    parser.add_argument('--input-size', type=int, default=448,
                        help='Input image size for model')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')

    # Patch arguments
    parser.add_argument('--patch-size', type=int, default=505,
                        help='Size of patches to extract (default: 505)')
    parser.add_argument('--lambda', dest='lambda_param', type=float, default=5.0,
                        help='Defect emphasis parameter (default: 5.0)')

    # Flat field correction arguments
    parser.add_argument('--ffc-kernel-size', type=int, default=101,
                        help='Kernel size for flat field correction (default: 101)')
    parser.add_argument('--ffc-method', type=str, default='mean',
                        choices=['mean', 'gaussian', 'morphology'],
                        help='FFC method (default: mean)')
    parser.add_argument('--ffc-clip-factor', type=float, default=0.9,
                        help='FFC highlight clipping factor (default: 0.9)')

    # Output arguments
    parser.add_argument('--output-dir', type=str, default='patch_results',
                        help='Directory to save output files')
    parser.add_argument('--batch-size', type=int, default=9,
                        help='Batch size for inference')
    parser.add_argument('--no-visualizations', action='store_true',
                        help='Skip generating visualization files')

    args = parser.parse_args()

    # Validate input
    if not args.image and not args.image_dir:
        parser.error("Please specify --image or --image-dir")

    # Setup device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"

    print("=" * 60)
    print("Patch-Based CrowdBT Inference Pipeline")
    print("=" * 60)

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
    except Exception:
        backbone = args.backbone
        input_size = args.input_size

    model = load_model(args.checkpoint, backbone_name=backbone, device=device)
    print(f"Model loaded: {backbone}, input size: {input_size}")

    # Load calibrator
    print(f"\nLoading calibrator from {args.calibrator}")
    calibrator = ScoreCalibrator.load(args.calibrator)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Process images
    save_viz = not args.no_visualizations

    print(f"Preprocessing: Grayscale + Flat Field Correction (kernel={args.ffc_kernel_size}, method={args.ffc_method})")

    if args.image:
        # Single image mode
        print(f"\nProcessing single image: {args.image}")
        result = process_single_image(
            model, calibrator, args.image,
            patch_size=args.patch_size,
            lambda_param=args.lambda_param,
            input_size=input_size,
            device=device,
            output_dir=args.output_dir,
            save_visualizations=save_viz,
            ffc_kernel_size=args.ffc_kernel_size,
            ffc_method=args.ffc_method,
            ffc_clip_factor=args.ffc_clip_factor
        )

        # Print results
        print("\n" + "=" * 60)
        print("Results")
        print("=" * 60)
        print(f"Image: {args.image}")
        print(f"Final Score: {result.final_score:.2f} ({get_quality_tier(result.final_score)})")
        print(f"Mean Patch Score: {result.mean_patch_score:.2f}")
        print(f"Min Patch Score: {result.min_patch_score:.2f}")
        print(f"Max Patch Score: {result.max_patch_score:.2f}")
        print(f"Std Dev: {result.std_patch_score:.2f}")
        print(f"\nPatch Scores:")
        for patch in result.patches:
            print(f"  Row {patch.row}, Col {patch.col}: {patch.calibrated_score:.2f}")

    else:
        # Directory mode
        print(f"\nProcessing images from: {args.image_dir}")
        results = process_image_directory(
            model, calibrator, args.image_dir,
            patch_size=args.patch_size,
            lambda_param=args.lambda_param,
            input_size=input_size,
            device=device,
            output_dir=args.output_dir,
            save_visualizations=save_viz,
            ffc_kernel_size=args.ffc_kernel_size,
            ffc_method=args.ffc_method,
            ffc_clip_factor=args.ffc_clip_factor
        )

        # Print summary
        if results:
            print("\n" + "=" * 60)
            print("Summary")
            print("=" * 60)
            final_scores = [r.final_score for r in results]
            print(f"Total images processed: {len(results)}")
            print(f"Mean final score: {np.mean(final_scores):.2f}")
            print(f"Min final score: {np.min(final_scores):.2f}")
            print(f"Max final score: {np.max(final_scores):.2f}")

            # Quality distribution
            excellent = sum(1 for s in final_scores if s >= 75)
            good = sum(1 for s in final_scores if 50 <= s < 75)
            fair = sum(1 for s in final_scores if 25 <= s < 50)
            poor = sum(1 for s in final_scores if s < 25)

            print(f"\nQuality Distribution:")
            print(f"  Excellent (75-100): {excellent} ({100*excellent/len(final_scores):.1f}%)")
            print(f"  Good (50-75):       {good} ({100*good/len(final_scores):.1f}%)")
            print(f"  Fair (25-50):       {fair} ({100*fair/len(final_scores):.1f}%)")
            print(f"  Poor (0-25):        {poor} ({100*poor/len(final_scores):.1f}%)")

    print("\n" + "=" * 60)
    print("Output Files")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    if save_viz:
        print("Generated for each image:")
        print("  - {name}_heatmap.png  - 3x3 grid heatmap")
        print("  - {name}_overlay.png  - Image with score overlay")
        print("  - {name}_summary.png  - Final score + statistics")
        print("  - {name}_result.json  - Machine-readable results")


if __name__ == '__main__':
    main()
