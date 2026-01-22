"""
Crowd-BT Loss function for learning from pairwise comparisons.

The Crowd-BT model extends Bradley-Terry to handle:
- Noisy annotations from multiple annotators
- Learnable annotator reliability (eta)
- Weighted comparisons based on confidence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class CrowdBTLoss(nn.Module):
    """
    Crowd Bradley-Terry loss for pairwise comparisons.

    The Bradley-Terry model defines:
        P(i > j) = sigma(s_i - s_j)

    where s_i is the quality score of item i and sigma is the sigmoid function.

    With annotator noise (eta parameter):
        P(annotator says i > j) = eta * P(i > j) + (1 - eta) * 0.5

    where eta in [0.5, 1] represents annotator reliability.
    eta = 1.0 means perfect annotator, eta = 0.5 means random guessing.
    """

    def __init__(
        self,
        eta_init: float = 0.8,
        eta_learnable: bool = True,
        margin: float = 0.0,
        draw_margin: float = 0.1
    ):
        """
        Args:
            eta_init: Initial annotator reliability (0.5 to 1.0)
            eta_learnable: Whether eta is a learnable parameter
            margin: Minimum margin for clear wins (optional)
            draw_margin: Score difference threshold for draws
        """
        super().__init__()

        # Annotator reliability parameter
        # Use sigmoid to constrain eta to [0.5, 1.0]
        # eta = 0.5 + 0.5 * sigmoid(eta_logit)
        eta_logit = self._eta_to_logit(eta_init)

        if eta_learnable:
            self.eta_logit = nn.Parameter(torch.tensor(eta_logit))
        else:
            self.register_buffer('eta_logit', torch.tensor(eta_logit))

        self.margin = margin
        self.draw_margin = draw_margin

    def _eta_to_logit(self, eta: float) -> float:
        """Convert eta [0.5, 1.0] to logit space."""
        # eta = 0.5 + 0.5 * sigmoid(logit)
        # sigmoid(logit) = (eta - 0.5) / 0.5 = 2 * eta - 1
        # logit = sigmoid_inv(2 * eta - 1)
        eta = max(0.501, min(0.999, eta))  # Clip for numerical stability
        sigmoid_val = 2 * eta - 1
        logit = torch.logit(torch.tensor(sigmoid_val)).item()
        return logit

    @property
    def eta(self) -> torch.Tensor:
        """Current annotator reliability value."""
        return 0.5 + 0.5 * torch.sigmoid(self.eta_logit)

    def forward(
        self,
        score_diff: torch.Tensor,
        labels: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the Crowd-BT loss.

        Args:
            score_diff: Score difference (s1 - s2) [B, 1] or [B]
            labels: Comparison labels [B]
                1 = img1 > img2
                -1 = img1 < img2
                0 = draw
            weights: Optional comparison weights [B]

        Returns:
            Dict with 'loss', 'accuracy', 'eta'
        """
        score_diff = score_diff.squeeze(-1)  # [B]
        labels = labels.float()

        if weights is None:
            weights = torch.ones_like(labels)

        batch_size = labels.shape[0]
        eta = self.eta

        # Separate handling for different label types
        win_mask = (labels == 1)  # img1 > img2
        lose_mask = (labels == -1)  # img1 < img2
        draw_mask = (labels == 0)

        loss = torch.zeros(1, device=score_diff.device)

        # For wins (label = 1): maximize P(img1 > img2)
        if win_mask.any():
            win_diff = score_diff[win_mask]
            win_weights = weights[win_mask]

            # P(img1 > img2) with noise
            p_win = eta * torch.sigmoid(win_diff) + (1 - eta) * 0.5
            loss_win = -torch.log(p_win + 1e-8)
            loss = loss + (loss_win * win_weights).sum()

        # For losses (label = -1): maximize P(img2 > img1) = P(img1 < img2)
        if lose_mask.any():
            lose_diff = score_diff[lose_mask]
            lose_weights = weights[lose_mask]

            # P(img2 > img1) with noise
            p_lose = eta * torch.sigmoid(-lose_diff) + (1 - eta) * 0.5
            loss_lose = -torch.log(p_lose + 1e-8)
            loss = loss + (loss_lose * lose_weights).sum()

        # For draws (label = 0): scores should be close
        if draw_mask.any():
            draw_diff = score_diff[draw_mask]
            draw_weights = weights[draw_mask]

            # Penalize large score differences for draws
            # Use a soft penalty that increases with |diff|
            loss_draw = draw_diff.abs()  # L1 penalty
            loss = loss + (loss_draw * draw_weights).sum() * 0.5

        # Normalize by total weight
        total_weight = weights.sum()
        loss = loss / (total_weight + 1e-8)

        # Compute accuracy
        with torch.no_grad():
            # Predictions based on score difference
            pred = torch.zeros_like(labels)
            pred[score_diff > self.draw_margin] = 1
            pred[score_diff < -self.draw_margin] = -1

            correct = (pred == labels).float()
            accuracy = (correct * weights).sum() / (total_weight + 1e-8)

        return {
            'loss': loss,
            'accuracy': accuracy,
            'eta': eta.detach(),
        }


class MarginRankingLoss(nn.Module):
    """
    Simple margin ranking loss for pairwise comparisons.

    Simpler alternative to Crowd-BT that doesn't model annotator noise.
    """

    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin
        self.loss_fn = nn.MarginRankingLoss(margin=margin, reduction='none')

    def forward(
        self,
        score_diff: torch.Tensor,
        labels: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute margin ranking loss.

        Args:
            score_diff: Score difference (s1 - s2) [B, 1] or [B]
            labels: Comparison labels (1, -1, or 0) [B]
            weights: Optional weights [B]

        Returns:
            Dict with 'loss', 'accuracy'
        """
        score_diff = score_diff.squeeze(-1)

        if weights is None:
            weights = torch.ones_like(labels)

        # Filter out draws for margin ranking
        non_draw = labels != 0
        if non_draw.any():
            diff = score_diff[non_draw]
            target = labels[non_draw]
            w = weights[non_draw]

            # MarginRankingLoss expects score1, score2, target
            # We have diff = score1 - score2
            # Create dummy scores
            score1 = diff / 2
            score2 = -diff / 2

            loss_vals = self.loss_fn(score1, score2, target)
            loss = (loss_vals * w).sum() / (w.sum() + 1e-8)
        else:
            loss = torch.tensor(0.0, device=score_diff.device)

        # Handle draws with L1 loss
        draw_mask = labels == 0
        if draw_mask.any():
            draw_diff = score_diff[draw_mask]
            draw_w = weights[draw_mask]
            draw_loss = (draw_diff.abs() * draw_w).sum() / (draw_w.sum() + 1e-8)
            loss = loss + 0.5 * draw_loss

        # Accuracy
        with torch.no_grad():
            pred = torch.sign(score_diff)
            pred[score_diff.abs() < 0.1] = 0  # Predict draw for small differences
            correct = (pred == labels).float()
            accuracy = (correct * weights).sum() / (weights.sum() + 1e-8)

        return {
            'loss': loss,
            'accuracy': accuracy,
        }


def get_loss_function(
    loss_type: str = "crowd_bt",
    **kwargs
) -> nn.Module:
    """
    Factory function to create loss functions.

    Args:
        loss_type: 'crowd_bt' or 'margin'
        **kwargs: Arguments passed to loss function

    Returns:
        Loss module
    """
    if loss_type == "crowd_bt":
        return CrowdBTLoss(**kwargs)
    elif loss_type == "margin":
        return MarginRankingLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
