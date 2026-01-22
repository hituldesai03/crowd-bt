"""
Configuration and default hyperparameters for Crowd-BT Quality Scorer.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import yaml
import os


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    # Model
    backbone_name: str = "efficientnet_b4"
    input_size: int = 448
    pretrained: bool = True

    # Training
    batch_size: int = 4
    epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5

    # Crowd-BT specific
    eta_init: float = 0.5  # Initial annotator reliability (learnable)
    eta_learnable: bool = True

    # Data
    train_split: float = 0.8
    num_workers: int = 4

    # Checkpointing
    checkpoint_dir: str = "results"
    save_every: int = 5

    # Logging
    log_interval: int = 10

    # Device
    device: str = "cuda"


@dataclass
class DataConfig:
    """Data source configuration."""
    # S3 settings
    bucket_name: str = "surface-quality-dataset"
    data_prefix: str = "quack_v2_data_user_logs/"

    # Local paths
    local_data_dir: str = "data"
    image_dir: str = "data/iter_0"

    # Comparison data files
    gold_batch_file: str = "data/gold_batch.json"
    anchor_anchor_file: str = "data/Anchor_Anchor.json"
    anchor_regular_file: str = "data/Anchor_Regular.json"
    regular_regular_file: str = "data/Regular_Regular.json"

    # Quality hierarchy (higher = better)
    quality_ranks: dict = field(default_factory=lambda: {
        'goldensample': 100,
        'POS_5': 5,
        'POS_4': 4,
        'POS_3': 3,
        'POS_2': 2,
        'POS_1': 1,
        'POS_0': 0,
        'spraying_POS_0': 0,
        'sanding_POS_0': 0,
    })


@dataclass
class Config:
    """Combined configuration."""
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load config from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        config = cls()

        if 'training' in data:
            for key, value in data['training'].items():
                if hasattr(config.training, key):
                    setattr(config.training, key, value)

        if 'data' in data:
            for key, value in data['data'].items():
                if hasattr(config.data, key):
                    setattr(config.data, key, value)

        return config

    def to_yaml(self, path: str):
        """Save config to YAML file."""
        data = {
            'training': {
                'backbone_name': self.training.backbone_name,
                'input_size': self.training.input_size,
                'pretrained': self.training.pretrained,
                'batch_size': self.training.batch_size,
                'epochs': self.training.epochs,
                'learning_rate': self.training.learning_rate,
                'weight_decay': self.training.weight_decay,
                'eta_init': self.training.eta_init,
                'eta_learnable': self.training.eta_learnable,
                'train_split': self.training.train_split,
                'num_workers': self.training.num_workers,
                'checkpoint_dir': self.training.checkpoint_dir,
                'save_every': self.training.save_every,
                'log_interval': self.training.log_interval,
                'device': self.training.device,
            },
            'data': {
                'bucket_name': self.data.bucket_name,
                'data_prefix': self.data.data_prefix,
                'local_data_dir': self.data.local_data_dir,
                'image_dir': self.data.image_dir,
            }
        }

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)


# Default config instance
default_config = Config()


def get_quality_rank(category: str, quality_ranks: dict = None) -> int:
    """
    Get quality rank from category string.
    Higher rank = better quality.
    """
    if quality_ranks is None:
        quality_ranks = default_config.data.quality_ranks

    if not category:
        return -1

    # Check for exact match first
    if category in quality_ranks:
        return quality_ranks[category]

    # Check for partial match (e.g., 'sanding_POS_5' contains 'POS_5')
    for key, rank in quality_ranks.items():
        if key in category:
            return rank

    return -1
