from dataclasses import dataclass, field
from typing import Union, List, Optional
import numpy as np

@dataclass
class EvalStepConfig:
    enabled: bool = False
    print_results: bool = True
    plot_results: bool = True
    folds: Union[str, List[int]] = "all"  # "all", "each", or specific folds like [1, 2, 3]

@dataclass
class EvaluationConfig:
    loss_curves: EvalStepConfig = field(default_factory=lambda: EvalStepConfig(enabled=True))
    metrics: EvalStepConfig = field(default_factory=lambda: EvalStepConfig(enabled=True))
    confidence: EvalStepConfig = field(default_factory=lambda: EvalStepConfig(enabled=True))
    confusion_matrices: EvalStepConfig = field(default_factory=lambda: EvalStepConfig(enabled=True))
    performance_by_count: EvalStepConfig = field(default_factory=lambda: EvalStepConfig(enabled=True))
    
    # Voxel-specific workflow settings
    voxel_cluster_parity: EvalStepConfig = field(default_factory=lambda: EvalStepConfig(enabled=False))
    voxel_threshold: float = 0.5
    voxel_k_threshold: float = 0.0
    voxel_top_k: Optional[int] = None
    voxel_min_cluster_voxels: int = 1
    voxel_size_mm: tuple = (22.9, 22.9, 31.95)
    voxel_mask: Optional[np.ndarray] = None
    
    # Postprocessing settings (BBox detection)
    postprocessing_method: str = "wbf"  # "wbf", "nms", or "none"
    top_k: Optional[int] = 3            # Max sources to retain (e.g. 3)
    k_threshold: float = 0.0            # Relative confidence ratio (0.0 to disable, or e.g. 0.35)
    conf_threshold: float = 0.35        # Minimum absolute confidence score threshold
    wbf_dist_threshold: float = 8.0     # 3D Euclidean distance (in voxels) for clustering & fusing boxes
    iou_threshold: float = 0.1          # 3D DIoU threshold for classic NMS fallback
    
    # Global settings
    n_folds: int = 5
    file_label: str = "res"
    label_mode: str = "bbox"
    match_mode: str = "hybrid"
    verbose: bool = True
