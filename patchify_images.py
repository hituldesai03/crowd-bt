"""
Patchify full images into patches matching the training data format.

This script:
1. Loads full images from a source directory
2. Applies flat field correction (FFC) - same as training preprocessing
3. Extracts non-overlapping patches (e.g., 505x505 from 1516x1516)
4. Saves patches as grayscale PNGs (matching training data format)
"""

import argparse
import os
import glob
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


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


def extract_patches(
    image: np.ndarray,
    patch_size: int = 505
) -> List[Tuple[np.ndarray, int, int, int, int]]:
    """
    Extract non-overlapping patches from image.

    Args:
        image: Grayscale image (H, W) as numpy array
        patch_size: Size of each patch (default 505)

    Returns:
        List of (patch, row, col, x, y) tuples
    """
    height, width = image.shape[:2]

    # Calculate number of patches in each dimension
    n_cols = width // patch_size
    n_rows = height // patch_size

    # Handle edge pixels
    extra_x = width - (n_cols * patch_size)
    extra_y = height - (n_rows * patch_size)

    patches = []

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
            x = max(0, x)
            y = max(0, y)

            # Extract patch
            patch = image[y:y + patch_size, x:x + patch_size]

            # Pad if needed (for images smaller than patch_size)
            if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                padded = np.zeros((patch_size, patch_size), dtype=np.uint8)
                padded[:patch.shape[0], :patch.shape[1]] = patch
                patch = padded

            patches.append((patch, row, col, x, y))

    return patches


def process_image(
    image_path: str,
    output_dir: str,
    patch_size: int = 505,
    ffc_kernel_size: int = 101,
    ffc_method: str = 'mean',
    ffc_clip_factor: float = 0.9
) -> List[str]:
    """
    Process a single image: apply FFC and extract patches.

    Args:
        image_path: Path to input image
        output_dir: Directory to save patches
        patch_size: Size of patches to extract
        ffc_kernel_size: Kernel size for flat field correction
        ffc_method: FFC method
        ffc_clip_factor: FFC clip factor

    Returns:
        List of saved patch file paths
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    img_array = np.array(image)

    # Apply flat field correction
    ffc_gray = compute_flat_field_corrected(
        img_array,
        kernel_size=ffc_kernel_size,
        method=ffc_method,
        clip_factor=ffc_clip_factor
    )

    # Extract patches
    patches = extract_patches(ffc_gray, patch_size)

    # Get base filename without extension
    basename = os.path.splitext(os.path.basename(image_path))[0]

    # Save patches
    saved_paths = []
    for patch, row, col, x, y in patches:
        patch_id = row * 3 + col  # 0-8 for 3x3 grid
        patch_filename = f"{basename}_patch_{patch_id}.png"
        patch_path = os.path.join(output_dir, patch_filename)

        # Save as grayscale PNG (matching training data format)
        Image.fromarray(patch, mode='L').save(patch_path)
        saved_paths.append(patch_path)

    return saved_paths


def main():
    parser = argparse.ArgumentParser(
        description='Patchify full images with FFC preprocessing'
    )

    parser.add_argument('--input-dir', type=str, default='full_image_data',
                        help='Directory containing full images')
    parser.add_argument('--output-dir', type=str, default='patchified_full_image_data',
                        help='Directory to save patches')
    parser.add_argument('--patch-size', type=int, default=505,
                        help='Size of patches to extract (default: 505)')

    # FFC arguments
    parser.add_argument('--ffc-kernel-size', type=int, default=101,
                        help='Kernel size for flat field correction (default: 101)')
    parser.add_argument('--ffc-method', type=str, default='mean',
                        choices=['mean', 'gaussian', 'morphology'],
                        help='FFC method (default: mean)')
    parser.add_argument('--ffc-clip-factor', type=float, default=0.9,
                        help='FFC highlight clipping factor (default: 0.9)')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Find all images
    image_paths = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
        image_paths.extend(glob.glob(os.path.join(args.input_dir, ext)))

    if not image_paths:
        print(f"No images found in {args.input_dir}")
        return

    print("=" * 60)
    print("Patchify Images with FFC Preprocessing")
    print("=" * 60)
    print(f"Input directory:  {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Patch size:       {args.patch_size}x{args.patch_size}")
    print(f"FFC kernel size:  {args.ffc_kernel_size}")
    print(f"FFC method:       {args.ffc_method}")
    print(f"FFC clip factor:  {args.ffc_clip_factor}")
    print(f"Images found:     {len(image_paths)}")
    print()

    # Process all images
    total_patches = 0
    for image_path in tqdm(image_paths, desc="Processing images"):
        try:
            saved_paths = process_image(
                image_path,
                args.output_dir,
                patch_size=args.patch_size,
                ffc_kernel_size=args.ffc_kernel_size,
                ffc_method=args.ffc_method,
                ffc_clip_factor=args.ffc_clip_factor
            )
            total_patches += len(saved_paths)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    print()
    print("=" * 60)
    print("Complete!")
    print("=" * 60)
    print(f"Total patches saved: {total_patches}")
    print(f"Output directory:    {args.output_dir}")


if __name__ == '__main__':
    main()
