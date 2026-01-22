"""
Quality Scorer model with swappable backbone.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from backbones import BackboneWrapper, get_recommended_input_size


class QualityScorer(nn.Module):
    """
    Neural network that scores image quality.

    Architecture:
    - Backbone (e.g., EfficientNet) extracts features
    - MLP head maps features to a single quality score

    The model outputs a scalar quality score for each input image.
    For pairwise comparisons, we compute score(img1) - score(img2).
    """

    def __init__(
        self,
        backbone_name: str = "efficientnet_b4",
        pretrained: bool = True,
        hidden_dim: int = 512,
        dropout: float = 0.2
    ):
        """
        Args:
            backbone_name: Name of the timm backbone
            pretrained: Whether to use pretrained backbone weights
            hidden_dim: Hidden dimension in the scoring head
            dropout: Dropout rate
        """
        super().__init__()

        self.backbone_name = backbone_name
        self.backbone = BackboneWrapper(backbone_name, pretrained)

        # Scoring head: features -> quality score
        self.head = nn.Sequential(
            nn.Linear(self.backbone.output_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Initialize head weights
        self._init_head()

    def _init_head(self):
        """Initialize the scoring head weights."""
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input images [B, 3, H, W]

        Returns:
            Quality scores [B, 1]
        """
        features = self.backbone(x)
        score = self.head(features)
        return score

    def score_pair(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Score a pair of images and return the difference.

        Args:
            img1: First image batch [B, 3, H, W]
            img2: Second image batch [B, 3, H, W]

        Returns:
            Score difference (score1 - score2) [B, 1]
        """
        score1 = self.forward(img1)
        score2 = self.forward(img2)
        return score1 - score2

    def get_recommended_input_size(self) -> int:
        """Get recommended input size for this backbone."""
        return get_recommended_input_size(self.backbone_name)

    @property
    def feature_dim(self) -> int:
        """Return the backbone feature dimension."""
        return self.backbone.output_dim


class PairwiseScorer(nn.Module):
    """
    Wrapper that handles pairwise comparisons.

    This module wraps QualityScorer and provides utilities for
    training with pairwise comparison data.
    """

    def __init__(
        self,
        backbone_name: str = "efficientnet_b4",
        pretrained: bool = True,
        **kwargs
    ):
        super().__init__()
        self.scorer = QualityScorer(backbone_name, pretrained, **kwargs)

    def forward(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for a pair of images.

        Args:
            img1: First image batch [B, 3, H, W]
            img2: Second image batch [B, 3, H, W]

        Returns:
            Dict with 'score1', 'score2', 'diff'
        """
        score1 = self.scorer(img1)
        score2 = self.scorer(img2)
        diff = score1 - score2

        return {
            'score1': score1,
            'score2': score2,
            'diff': diff
        }

    def score(self, img: torch.Tensor) -> torch.Tensor:
        """Score a single image."""
        return self.scorer(img)


def load_model(
    checkpoint_path: str,
    backbone_name: str = "efficientnet_b4",
    device: str = "cuda"
) -> QualityScorer:
    """
    Load a trained model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        backbone_name: Backbone architecture
        device: Device to load model on

    Returns:
        Loaded QualityScorer model
    """
    model = QualityScorer(backbone_name=backbone_name, pretrained=False)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model
