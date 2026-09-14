from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple
import numpy as np

@dataclass
class PLACENetConfig:
    """Configuration for PLACENet model."""
    pad_value: int = -1 # Value to pad the empty slots with
    l2_reg: float = 0.01 # L2 regularization parameter
    learning_rate: float = 1e-4 # Learning rate (fixed for now)
    warmup_epochs: int = 0 # Number of epochs to linearly warmup the learning rate
    loss_type: str = "hybrid"  # "smooth_l1", "ciou", or "hybrid"
    epochs: int = 3000 # Number of epochs to train for
    batch_size: int = 32 # Batch size
    folds: int = 5 # Number of folds for cross-validation
    max_sources: int = 1 # Maximum number of sources to classify
    run_name: str = "PLACENet_run" # Name of the run for loggingss
    matching_strategy: str = "jv" # "greedy" or "jv"
    delta: float = 1.0  # For smooth_l1
    ciou_weight: float = 1.0 # Weight for the CIoU loss
    smooth_weight: float = 1.0 # Weight for the smooth L1 loss
    confidence_weight: float = 1.5 # Weight for the confidence loss
    noobj_weight: float = 0.1 # Weight for the no object loss
    enable_augmentation: bool = False # If True, enable data augmentation
    augmentation_multiplier: int = 0 # How many augmented copies to create per original sample
    augmenter_kwargs: Dict[str, Any] = field(default_factory=dict) # Keyword arguments for the augmenter
    early_stopping_monitor: str = "val_iou"  # "val_iou" (maximize) or "val_loss" (minimize)
    early_stopping_patience: int = 30
    val_iou_confidence_threshold: float = 0.5  # conf threshold when computing val IoU for early stopping
    backbone: str = "cnn"  # "cnn" (2D conv on detector×bin image) or "transformer" (per-detector Conv1D + MHA)
    transformer_num_heads: int = 4
    transformer_key_dim: int = 32
    transformer_conv_filters: Tuple[int, int] = (64, 32)  # two TimeDistributed Conv1D stages
    label_mode: Literal["bbox", "position", "voxel"] = "bbox"  # bbox / position / coarse occupancy
    voxel_grid: Tuple[int, int, int] = (10, 10, 10)
    voxel_mask: Optional[np.ndarray] = field(default=None, repr=False)
    voxel_focal_gamma: float = 2.0
    voxel_focal_alpha: float = 0.25
    voxel_mask_loss: bool = True
    # Per-axis multipliers on smooth-L1 terms for position mode (x, y, z).
    position_axis_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0)


@dataclass
class PLACENetTrainingResult:
    """Result of training the PLACENet model."""
    run_label: str # Name of the run
    metrics: List[Dict[str, float]] # Metrics for the run
    histories: List[Any] # Histories for the run
    dataset_name: Optional[str] = None # Name of the dataset
