"""
PyTorch Dataset for pairwise image comparisons.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from typing import List, Dict, Optional, Tuple
import random

from config import default_config


def get_transforms(input_size: int, is_train: bool = True) -> transforms.Compose:
    """
    Get image transforms for training or evaluation.

    Args:
        input_size: Target image size
        is_train: Whether to apply training augmentations

    Returns:
        torchvision transforms composition
    """
    # ImageNet normalization
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    if is_train:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            normalize,
        ])


class PairwiseComparisonDataset(Dataset):
    """
    Dataset for pairwise image comparisons.

    Each sample contains:
    - img1: First image tensor
    - img2: Second image tensor
    - label: 1 if img1 > img2, -1 if img1 < img2, 0 if draw
    - weight: Confidence weight for this comparison
    """

    def __init__(
        self,
        comparisons: List[Dict],
        image_dir: str = None,
        input_size: int = 448,
        is_train: bool = True,
        transform: Optional[transforms.Compose] = None
    ):
        """
        Args:
            comparisons: List of comparison dicts with 'img1', 'img2', 'label', 'weight'
            image_dir: Directory containing images
            input_size: Target image size
            is_train: Whether this is training data
            transform: Optional custom transform
        """
        self.comparisons = comparisons
        self.image_dir = image_dir or default_config.data.image_dir
        self.input_size = input_size
        self.is_train = is_train
        self.transform = transform or get_transforms(input_size, is_train)

        # Validate that images exist
        self._validate_images()

    def _validate_images(self):
        """Check that all referenced images exist."""
        missing = set()
        for comp in self.comparisons:
            for img_key in ['img1', 'img2']:
                img_path = os.path.join(self.image_dir, comp[img_key])
                if not os.path.exists(img_path):
                    missing.add(comp[img_key])

        if missing:
            print(f"Warning: {len(missing)} images not found in {self.image_dir}")
            # Filter out comparisons with missing images
            valid_comparisons = []
            for comp in self.comparisons:
                img1_path = os.path.join(self.image_dir, comp['img1'])
                img2_path = os.path.join(self.image_dir, comp['img2'])
                if os.path.exists(img1_path) and os.path.exists(img2_path):
                    valid_comparisons.append(comp)
            print(f"Kept {len(valid_comparisons)}/{len(self.comparisons)} valid comparisons")
            self.comparisons = valid_comparisons

    def __len__(self) -> int:
        return len(self.comparisons)

    def _load_image(self, filename: str) -> torch.Tensor:
        """Load and transform a single image."""
        img_path = os.path.join(self.image_dir, filename)
        image = Image.open(img_path).convert('RGB')
        return self.transform(image)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        comp = self.comparisons[idx]

        img1 = self._load_image(comp['img1'])
        img2 = self._load_image(comp['img2'])

        # Label: 1 if img1 > img2, -1 if img1 < img2, 0 if draw
        label = torch.tensor(comp['label'], dtype=torch.float32)

        # Weight for this comparison
        weight = torch.tensor(comp.get('weight', 1.0), dtype=torch.float32)

        return {
            'img1': img1,
            'img2': img2,
            'label': label,
            'weight': weight,
        }


class SingleImageDataset(Dataset):
    """
    Dataset for scoring individual images (inference mode).
    """

    def __init__(
        self,
        image_paths: List[str],
        input_size: int = 448,
        transform: Optional[transforms.Compose] = None
    ):
        """
        Args:
            image_paths: List of image file paths
            input_size: Target image size
            transform: Optional custom transform
        """
        self.image_paths = image_paths
        self.input_size = input_size
        self.transform = transform or get_transforms(input_size, is_train=False)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict:
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image_tensor = self.transform(image)

        return {
            'image': image_tensor,
            'path': img_path,
        }


def create_data_loaders(
    train_comparisons: List[Dict],
    val_comparisons: List[Dict],
    image_dir: str = None,
    input_size: int = 448,
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders.

    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = PairwiseComparisonDataset(
        train_comparisons,
        image_dir=image_dir,
        input_size=input_size,
        is_train=True
    )

    val_dataset = PairwiseComparisonDataset(
        val_comparisons,
        image_dir=image_dir,
        input_size=input_size,
        is_train=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


def split_comparisons(
    comparisons: List[Dict],
    train_ratio: float = 0.8,
    stratify_by_pair_type: bool = True,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split comparisons into train and validation sets.

    Args:
        comparisons: All comparisons
        train_ratio: Fraction for training
        stratify_by_pair_type: Whether to stratify by pair type
        seed: Random seed

    Returns:
        Tuple of (train_comparisons, val_comparisons)
    """
    random.seed(seed)

    if stratify_by_pair_type:
        # Group by pair type
        by_type = {}
        for comp in comparisons:
            pair_type = comp.get('pair_type', 'unknown')
            if pair_type not in by_type:
                by_type[pair_type] = []
            by_type[pair_type].append(comp)

        train_comps = []
        val_comps = []

        for pair_type, comps in by_type.items():
            random.shuffle(comps)
            split_idx = int(len(comps) * train_ratio)
            train_comps.extend(comps[:split_idx])
            val_comps.extend(comps[split_idx:])

        random.shuffle(train_comps)
        random.shuffle(val_comps)
    else:
        shuffled = comparisons.copy()
        random.shuffle(shuffled)
        split_idx = int(len(shuffled) * train_ratio)
        train_comps = shuffled[:split_idx]
        val_comps = shuffled[split_idx:]

    print(f"Split: {len(train_comps)} train, {len(val_comps)} val")
    return train_comps, val_comps
