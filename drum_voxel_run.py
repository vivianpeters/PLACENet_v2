"""
Clean run script for PLACENet using the original Drum Geometry in Voxel mode.
This script demonstrates the end-to-end pipeline: data prep, training, and evaluation.
"""

from pathlib import Path
import numpy as np

from preprocessing.load_data import PLACENetPrep, PLACEPrepConfig
from model.model_config import PLACENetConfig
from model.train import PLACENet
from evaluation.evaluator import Evaluator
from evaluation.eval_config import EvaluationConfig, EvalStepConfig
from auxiliaries.geometry import PLACEGeometry, fine_voxel_to_coarse_indices, coarse_inner_drum_mask
from plot.plot_loss_curves import save_loss_curves

import random

# ---------------------------------------------------------
# 1. Configuration & Drum Geometry Setup
# ---------------------------------------------------------
data_root = "/your/csv/data/path"
run_name = "your-run-name"

# Initialize the physical drum geometry
drum = PLACEGeometry()

# ---------------------------------------------------------
# 2. Data Preparation
# ---------------------------------------------------------
prep = PLACENetPrep(
    PLACEPrepConfig(
        data_dir=data_root,
        dets=56,          
        chunk_size=56,    
        max_sources=3,    
        label_mode="voxel",
        voxel_grid=(10, 10, 10),
        coordinate_mapper=lambda x, y, z, shape: fine_voxel_to_coarse_indices(x, y, z, drum, shape),
        cluster_method="hdbscan",
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
# Get the voxel mask from the geometry to restrict predictions to physical space
mask = coarse_inner_drum_mask(drum, (10, 10, 10))

trainer = PLACENet(
    PLACENetConfig(
        run_name=run_name,
        max_sources=3,
        label_mode="voxel",
        voxel_grid=(10, 10, 10),
        voxel_mask=mask,
        
        # Model Architecture & Hyperparameters
        backbone="cnn",
        learning_rate=1e-4,
        warmup_epochs=20,
        l2_reg=1e-3,
        epochs=3000,
        batch_size=32,
        folds=5,
        
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

# Set up evaluation config for voxels
eval_config = EvaluationConfig(
    label_mode="voxel",
    file_label=run_name,
    voxel_threshold=0.5,
    voxel_cluster_parity=EvalStepConfig(enabled=True, plot_results=True, print_results=True),
    loss_curves=EvalStepConfig(enabled=False),
    confusion_matrices=EvalStepConfig(enabled=True, plot_results=True, print_results=True),
    performance_by_count=EvalStepConfig(enabled=True, plot_results=True, print_results=True)
)

# Run the comprehensive evaluation suite
evaluator = Evaluator(eval_config, results_dir=str(res_dir), data_dir=data_root)
evaluator.run() 

print("\nEvaluation complete! Check the Results folder for plots and metrics.")

# Plot sample predictions
try:
    from plot.plot_samples import PLACENetPlot
    plotter = PLACENetPlot()
    print("\nGenerating drum sample plot...")
    plotter.plot_voxel_predictions(res_dir, file_label=run_name, sample_idx=0, threshold=0.5)
except Exception as e:
    print(f"\nCould not generate sample plot: {e}")
