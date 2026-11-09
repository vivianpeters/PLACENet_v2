"""
Clean run script for PLACENet using the original Drum Geometry in Bounding Box mode.
This script demonstrates the end-to-end pipeline: data prep, training, and evaluation.
"""

from pathlib import Path

from preprocessing.load_data import PLACENetPrep, PLACEPrepConfig
from model.model_config import PLACENetConfig
from model.train import PLACENet
from evaluation.evaluator import Evaluator
from evaluation.eval_config import EvaluationConfig, EvalStepConfig
from auxiliaries.geometry import PLACEGeometry
from plot.plot_loss_curves import save_loss_curves

import random

# ---------------------------------------------------------
# 1. Configuration & Drum Geometry Setup
# ---------------------------------------------------------
data_root = "/your/directory/"
run_name = "your_run_name"

# Initialize the physical drum geometry (used for plotting evaluation later)
drum = PLACEGeometry()

# ---------------------------------------------------------
# 2. Data Preparation
# ---------------------------------------------------------
prep = PLACENetPrep(
    PLACEPrepConfig(
        data_dir=data_root,
        dets=56,          # Number of detectors/units
        chunk_size=56,    # Number of spectra to group together
        max_sources=10,    # Maximum number of sources to classify
        label_mode="bbox",
        cluster_method="hdbscan",
        cluster_eps=2.0,
        cluster_min_cluster_size=5, 
        cluster_min_samples=3,  # HDBSCAN parameter
        cluster_merge_dist=6.0,  # Merge cluster centroids closer than 6 voxels (~14 cm)
        max_physical_sources=3,  # Cap at max 3 physical ground-truth sources
        randomize_slot_assignment=True,
    )
)

print(f"Looking for data in {data_root}...")
datasets = prep.load_datasets()

if not datasets:
    print(f"No data found! Update 'data_root' to point to your CSV files.")
else:
    print(f"Loaded {len(datasets)} dataset(s).")
    for dataset in datasets:
        print(dataset.summary())

# ---------------------------------------------------------
# 3. Model Training
# ---------------------------------------------------------
trainer = PLACENet(
    PLACENetConfig(
        run_name=run_name,
        max_sources=10,
        label_mode="bbox",
        
        # Model Architecture & Hyperparameters
        backbone="cnn",
        learning_rate=1e-4,
        warmup_epochs=20,
        l2_reg=1e-3,
        epochs=3000,
        batch_size=32,
        folds=5,
        
        # Bbox specific settings
        loss_type="hybrid",
        ciou_weight=1.0,
        smooth_weight=1.0,
        confidence_weight=1.5,
        noobj_weight=0.2,
        
        # Early stopping
        early_stopping_monitor="val_loss",
        early_stopping_patience=30,
        
        # Data augmentation
        enable_augmentation=False,
    )
)

print(f"\nStarting training for: {run_name}...")
result = trainer.train_concatenated(datasets)

print(f"\nTraining completed successfully!")
print(f"Final Metrics: {result.metrics[-1]}")

# ---------------------------------------------------------
# 4. Evaluation & Plotting
# ---------------------------------------------------------
print("\nRunning Evaluation...")
res_dir = Path(f"Results_{run_name}")

# Save the loss curves
print(f"Saving loss curves to {res_dir / 'loss_curves.pdf'}")
save_loss_curves(result.histories, res_dir / "loss_curves.pdf")

# Set up evaluation config
eval_config = EvaluationConfig(
    n_folds=5,
    file_label=run_name,
    label_mode="bbox",
    
    # Postprocessing settings
    postprocessing_method="wbf",      # "wbf", "nms", or "none"
    top_k=None,                          # Top-K source cap (at most 3 sources)
    k_threshold=0.35,                 # Relative confidence threshold ratio (conf >= k_threshold * max_conf)
    conf_threshold=0.35,              # Minimum absolute confidence score threshold
    wbf_dist_threshold=8.0,           # 3D Euclidean centroid distance (in voxels) for fusing adjacent boxes
    
    loss_curves=EvalStepConfig(enabled=False),
    confidence=EvalStepConfig(enabled=True, print_results=True, plot_results=True),
    confusion_matrices=EvalStepConfig(enabled=True, print_results=True, plot_results=True),
    performance_by_count=EvalStepConfig(enabled=True, print_results=True, plot_results=True)
)

# Run the evaluation
evaluator = Evaluator(eval_config, results_dir=str(res_dir), data_dir=data_root)
evaluator.run()

print("\nEvaluation complete! Check the Results folder for plots and metrics.")

# Plot sample predictions
try:
    from plot.plot_samples import PLACENetPlot
    plotter = PLACENetPlot()
    print("\nGenerating drum sample plot...")
    rand_idx = random.randint(0,1000)
    plotter.compare_histo(res_dir, file_label=run_name, sample_idx=rand_idx)
except Exception as e:
    print(f"\nCould not generate sample plot: {e}")