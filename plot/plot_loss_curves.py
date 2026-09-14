from typing import Sequence
from pathlib import Path
import matplotlib.pyplot as plt

def save_loss_curves(
    histories: Sequence,
    output_path: Path,
) -> None:
    """Save training vs validation loss curves to disk."""
    if not histories:
        print("No training histories available to plot.")
        return

    fig, ax_loss = plt.subplots(figsize=(8, 5))
    plotted = False
    import numpy as np
    import matplotlib.cm as cm
    
    blues = cm.Blues(np.linspace(0.4, 0.9, len(histories)))
    oranges = cm.Oranges(np.linspace(0.4, 0.9, len(histories)))

    for idx, history in enumerate(histories):
        loss = history.history.get("loss")
        val_loss = history.history.get("val_loss")
        if loss is None or val_loss is None:
            continue
        epochs = range(1, len(loss) + 1)
        ax_loss.plot(epochs, loss, label=f"Fold {idx+1} - Train", alpha=0.8, color=blues[idx], linewidth=1.5)
        ax_loss.plot(
            epochs, val_loss, label=f"Fold {idx+1} - Val loss", linestyle="--", alpha=0.8, color=oranges[idx], linewidth=1.5
        )
        plotted = True

    if not plotted:
        print("No loss curves found in histories.")
        plt.close()
        return

    ax_loss.set_xlabel("Epoch", fontsize=16)
    ax_loss.set_ylabel("Loss", fontsize=16)
    ax_loss.legend(fontsize=14)
    ax_loss.grid(True, linestyle="--", alpha=0.3)
    ax_loss.tick_params(axis="both", which="major", labelsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved loss curves to {output_path}")

def plot_loss_curves_from_tb(logs_dir: Path, output_path: Path):
    """Read TensorBoard logs from `logs_dir` and save the loss curves PDF."""
    if not logs_dir.exists():
        print(f"No logs directory found at {logs_dir}")
        return

    # Find all fold directories in logs_dir
    fold_dirs = sorted([d for d in logs_dir.iterdir() if d.is_dir()])
    
    # We may have multiple timestamps for the same fold, so group by fold index
    # Folder names look like: pu-240_drum_voxel20x20x20_cnn_kf0_20260820-175610
    
    # Let's just group by fold name (e.g. kf0, kf1)
    folds_data = {}
    
    import tensorflow as tf
    from tensorboard.backend.event_processing import event_accumulator
    
    for d in fold_dirs:
        # Extract fold identifier, e.g. "kf0"
        parts = d.name.split('_')
        kf = next((p for p in parts if p.startswith('kf')), None)
        if not kf:
            continue
            
        train_ea = event_accumulator.EventAccumulator(str(d / "train"), size_guidance={'tensors': 0})
        val_ea = event_accumulator.EventAccumulator(str(d / "validation"), size_guidance={'tensors': 0})
        
        try:
            train_ea.Reload()
            val_ea.Reload()
            
            # TensorBoard tags: 'epoch_loss', 'epoch_val_loss' 
            # (Note: Keras TensorBoard callback usually saves epoch_loss under 'epoch_loss' in both train and val)
            
            # For newer keras, train loss is 'epoch_loss' in train, val loss is 'epoch_loss' in validation
            try:
                train_loss_events = train_ea.Tensors("epoch_loss")
                val_loss_events = val_ea.Tensors("epoch_loss")
            except KeyError:
                print(f"Skipping {d.name} because 'epoch_loss' was not found.")
                continue
            
            train_loss = [tf.make_ndarray(e.tensor_proto).item() for e in train_loss_events]
            val_loss = [tf.make_ndarray(e.tensor_proto).item() for e in val_loss_events]
            
            # Keep the latest timestamp for each fold
            folds_data[kf] = {
                'loss': train_loss,
                'val_loss': val_loss
            }
        except Exception as e:
            print(f"Skipping {d.name} due to error: {e}")
            continue
            
    if not folds_data:
        print("No valid TensorBoard logs found to plot.")
        return
        
    fig, ax_loss = plt.subplots(figsize=(8, 5))
    
    import matplotlib.cm as cm
    import numpy as np
    blues = cm.Blues(np.linspace(0.4, 0.9, len(folds_data)))
    oranges = cm.Oranges(np.linspace(0.4, 0.9, len(folds_data)))
    
    for idx, (kf, data) in enumerate(sorted(folds_data.items())):
        loss = data['loss']
        val_loss = data['val_loss']
        
        epochs = range(1, len(loss) + 1)
        ax_loss.plot(epochs, loss, label=f"Fold {idx+1} - Train", alpha=0.8, color=blues[idx], linewidth=1.5)
        
        val_epochs = range(1, len(val_loss) + 1)
        ax_loss.plot(
            val_epochs, val_loss, label=f"Fold {idx+1} - Val loss", linestyle="--", alpha=0.8, color=oranges[idx], linewidth=1.5
        )
            
    ax_loss.set_xlabel("Epoch", fontsize=16)
    ax_loss.set_ylabel("Loss", fontsize=16)
    ax_loss.legend(fontsize=14)
    ax_loss.grid(True, linestyle="--", alpha=0.3)
    ax_loss.tick_params(axis="both", which="major", labelsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved loss curves from TensorBoard logs to {output_path}")