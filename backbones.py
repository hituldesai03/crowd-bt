"""
Backbone registry using timm for the Quality Scorer model.
"""

import timm
import torch.nn as nn
from typing import Dict, Tuple, Optional


# Registry of supported backbones with their output dimensions
BACKBONE_REGISTRY: Dict[str, Dict] = {
    # EfficientNet family
    'efficientnet_b0': {'features': 1280},
    'efficientnet_b1': {'features': 1280},
    'efficientnet_b2': {'features': 1408},
    'efficientnet_b3': {'features': 1536},
    'efficientnet_b4': {'features': 1792},
    'efficientnet_b5': {'features': 2048},
    'efficientnet_b6': {'features': 2304},
    'efficientnet_b7': {'features': 2560},

    # EfficientNetV2
    'efficientnetv2_s': {'features': 1280},
    'efficientnetv2_m': {'features': 1280},
    'efficientnetv2_l': {'features': 1280},

    # ResNet family
    'resnet18': {'features': 512},
    'resnet34': {'features': 512},
    'resnet50': {'features': 2048},
    'resnet101': {'features': 2048},
    'resnet152': {'features': 2048},

    # ConvNeXt
    'convnext_tiny': {'features': 768},
    'convnext_small': {'features': 768},
    'convnext_base': {'features': 1024},
    'convnext_large': {'features': 1536},

    # Vision Transformer (ViT)
    'vit_tiny_patch16_224': {'features': 192},
    'vit_small_patch16_224': {'features': 384},
    'vit_base_patch16_224': {'features': 768},
    'vit_base_patch16_384': {'features': 768},
    'vit_large_patch16_224': {'features': 1024},

    # Swin Transformer
    'swin_tiny_patch4_window7_224': {'features': 768},
    'swin_small_patch4_window7_224': {'features': 768},
    'swin_base_patch4_window7_224': {'features': 1024},

    # MobileNet
    'mobilenetv3_large_100': {'features': 1280},
    'mobilenetv3_small_100': {'features': 1024},
}


def get_backbone(name: str, pretrained: bool = True, scriptable: bool = False) -> Tuple[nn.Module, int]:
    """
    Create a backbone model using timm.

    Args:
        name: Name of the backbone (must be in BACKBONE_REGISTRY or a valid timm model)
        pretrained: Whether to load pretrained weights
        scriptable: If True, disables in-place operations for TorchScript/DDP compatibility

    Returns:
        Tuple of (backbone_model, feature_dimension)
    """
    # Create model without classifier head
    # scriptable=True disables in-place operations, making it compatible with DDP
    model = timm.create_model(name, pretrained=pretrained, num_classes=0, scriptable=scriptable)

    # Get feature dimension
    if name in BACKBONE_REGISTRY:
        feature_dim = BACKBONE_REGISTRY[name]['features']
    else:
        # Try to infer feature dimension from model
        feature_dim = model.num_features

    return model, feature_dim


def list_available_backbones() -> list:
    """List all registered backbones."""
    return list(BACKBONE_REGISTRY.keys())


def get_backbone_info(name: str) -> Optional[Dict]:
    """Get info about a backbone."""
    return BACKBONE_REGISTRY.get(name)


class BackboneWrapper(nn.Module):
    """
    Wrapper around timm backbone that ensures consistent output.
    """

    def __init__(self, name: str, pretrained: bool = True):
        super().__init__()
        self.backbone, self.feature_dim = get_backbone(name, pretrained)
        self.name = name

    def forward(self, x):
        """Extract features from input images."""
        return self.backbone(x)

    @property
    def output_dim(self) -> int:
        """Return the feature dimension."""
        return self.feature_dim


def get_recommended_input_size(name: str) -> int:
    """
    Get recommended input size for a backbone.
    """
    # ViT models have specific input size requirements
    if 'vit' in name.lower():
        if '384' in name:
            return 384
        return 224

    # Swin transformers
    if 'swin' in name.lower():
        return 224

    # EfficientNet recommended sizes
    efficientnet_sizes = {
        'efficientnet_b0': 224,
        'efficientnet_b1': 240,
        'efficientnet_b2': 260,
        'efficientnet_b3': 300,
        'efficientnet_b4': 380,
        'efficientnet_b5': 456,
        'efficientnet_b6': 528,
        'efficientnet_b7': 600,
    }

    if name in efficientnet_sizes:
        return efficientnet_sizes[name]

    # Default
    return 224
