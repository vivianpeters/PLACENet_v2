from dataclasses import dataclass, field
from typing import Union, List

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
    voxel_size_mm: tuple = (22.9, 22.9, 31.95)
    
    # Global settings
    n_folds: int = 5
    file_label: str = "res"
    label_mode: str = "bbox"
    match_mode: str = "hybrid"
    verbose: bool = True
