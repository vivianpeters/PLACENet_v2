import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

def plot_confidence_by_slot_type(
        pred_confidences: np.ndarray,
        true_confidences: np.ndarray,
        output_path: Path = None,
        title: Optional[str] = None,
    ):
        """
        Plot the distribution of predicted confidence scores separated by slot type (valid vs empty),
        with mean and median lines for each group.
        
        Args:
            pred_confidences: Array of predicted confidence scores (can be any shape, will be flattened)
            true_confidences: Array of true confidence scores (1.0 for valid slots, 0.0 for empty)
            output_path: Path to save the plot (if None, plot is not saved)
            title: Optional title for the plot
        """
        if pred_confidences is None or pred_confidences.size == 0:
            print("Warning: No confidence scores available for plotting.")
            return
        
        if true_confidences is None or true_confidences.size == 0:
            print("Warning: No true confidence labels available for plotting.")
            return
        
        # Flatten confidence arrays
        pred_conf_flat = pred_confidences.flatten()
        true_conf_flat = true_confidences.flatten()
        
        # Create masks
        valid_mask = true_conf_flat == 1.0
        empty_mask = true_conf_flat == 0.0
        
        if not np.any(valid_mask) and not np.any(empty_mask):
            print("Warning: No valid or empty slots found.")
            return
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot histograms
        if np.any(valid_mask):
            valid_conf = pred_conf_flat[valid_mask]
            ax.hist(valid_conf, bins=50, alpha=0.6, color='#2ca02c', 
                   label=f'Valid slots (n={np.sum(valid_mask)})', edgecolor='black', linewidth=0.5)
            # Add mean and median lines for valid slots
            valid_mean = valid_conf.mean()
            valid_median = np.median(valid_conf)
            ax.axvline(valid_mean, color='darkgreen', linestyle='--', linewidth=2.5, 
                      label=f'Valid mean: {valid_mean:.2f}')
            ax.axvline(valid_median, color='green', linestyle=':', linewidth=2.5, 
                      label=f'Valid median: {valid_median:.2f}')
        
        if np.any(empty_mask):
            empty_conf = pred_conf_flat[empty_mask]
            ax.hist(empty_conf, bins=50, alpha=0.6, color='#ff7f0e', 
                   label=f'Empty slots (n={np.sum(empty_mask)})', edgecolor='black', linewidth=0.5)
            # Add mean and median lines for empty slots
            empty_mean = empty_conf.mean()
            empty_median = np.median(empty_conf)
            ax.axvline(empty_mean, color='darkorange', linestyle='--', linewidth=2.5, 
                      label=f'Empty mean: {empty_mean:.2f}')
            ax.axvline(empty_median, color='orange', linestyle=':', linewidth=2.5, 
                      label=f'Empty median: {empty_median:.2f}')
        
        ax.set_xlabel('Predicted confidence score', fontsize=26)
        ax.set_ylabel('Frequency', fontsize=26)
        #if title:
        #    ax.set_title(title, fontsize=18, fontweight='bold')
        #else:
        #    ax.set_title('Distribution by Slot Type', fontsize=18, fontweight='bold')
        
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=26)

        # Apply scientific notation to y-axis
        import matplotlib.ticker as ticker
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((0, 0))
        ax.yaxis.set_major_formatter(formatter)
        ax.yaxis.get_offset_text().set_fontsize(26)
        
        # Increase tick label font sizes
        ax.tick_params(axis='both', which='major', labelsize=26)
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Set x-axis limits to start at 0 (after legend is placed)
        x_min = -0.05
        x_max = max(1.0, pred_conf_flat.max() + 0.05)
        ax.set_xlim(x_min, x_max)
        
        # Adjust layout to make room for legend on the right
        #plt.subplots_adjust(right=0.75)
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=200, bbox_inches='tight')
            plt.close()
            print(f"Saved confidence by slot type plot to {output_path}")
        else:
            plt.show()
            plt.close()