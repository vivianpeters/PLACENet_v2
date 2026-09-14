import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

def plot_pred_vs_gt_parity(
    y_true_all: np.ndarray,
    y_pred_all: np.ndarray,
    output_path: Path = None,
    label_mode: str = "bbox",
    confidence_threshold: float = 0.5,
    do_plot: bool = True,
    do_print: bool = True,
    subsample_plot: bool = True,
) -> None:
    """
    Generate grid of parity/scatter plots (Ground Truth vs Prediction) for box components,
    and optionally print the R2 and MAE metrics.
    """
    target_dim = 6 if label_mode == "bbox" else 3
    output_dim = target_dim + 1

    y_true_t = y_true_all[..., :target_dim]
    y_pred_t = y_pred_all[..., :target_dim]

    valid = ~np.all(y_true_t == 0.0, axis=-1) & ~np.all(y_pred_t == 0.0, axis=-1)
    
    # Filter parity scatter points by confidence threshold if confidence dimension exists
    if y_pred_all.shape[-1] >= output_dim and confidence_threshold > 0.0:
        confidences = y_pred_all[..., target_dim]
        valid = valid & (confidences >= confidence_threshold)

    if not np.any(valid):
        if do_print:
            print("  No valid slots for parity plot/metrics.")
        return

    comp_names = [
        ("x_center", "X Center (cm)", 1),
        ("y_center", "Y Center (cm)", 3),
        ("z_center", "Z Center (cm)", 5),
        ("x_width", "X Width (cm)", 0),
        ("y_width", "Y Width (cm)", 2),
        ("z_width", "Z Width (cm)", 4),
    ] if label_mode == "bbox" else [
        ("x_center", "X Center (cm)", 0),
        ("y_center", "Y Center (cm)", 1),
        ("z_center", "Z Center (cm)", 2),
    ]

    metrics = {}
    for i, (name, title, idx) in enumerate(comp_names):
        yt = y_true_t[valid, idx]
        yp = y_pred_t[valid, idx]
        if len(yt) > 0:
            metrics[name] = {
                "r2": r2_score(yt, yp),
                "mae": np.mean(np.abs(yt - yp)),
                "title": title,
                "yt": yt,
                "yp": yp
            }

    if do_print:
        print("\n--- Parity Metrics (True Positives) ---")
        for name, m in metrics.items():
            print(f"  {m['title']:15s}: R² = {m['r2']:.4f} | MAE = {m['mae']:.4f} cm")

    if do_plot and output_path:
        rows = 2 if label_mode == "bbox" else 1
        cols = 3
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5.5 * rows))
        
        # Make axes iterable flat even if 1D
        if rows == 1:
            axes = np.array(axes).flatten()
        else:
            axes = axes.flatten()

        for i, (name, title, idx) in enumerate(comp_names):
            ax = axes[i]
            
            if name not in metrics:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                continue
                
            m = metrics[name]
            yt = m["yt"]
            yp = m["yp"]
            r2 = m["r2"]
            mae = m["mae"]

            if subsample_plot and len(yt) > 3000:
                sub = np.random.choice(len(yt), 3000, replace=False)
                yt_plot, yp_plot = yt[sub], yp[sub]
            else:
                yt_plot, yp_plot = yt, yp

            ax.scatter(yt_plot, yp_plot, alpha=0.25, color="navy", s=10, edgecolors="none")

            min_val = min(yt.min(), yp.min())
            max_val = max(yt.max(), yp.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=1.5, label="Identity (y = x)")
            
            if len(yt) > 1:
                fit_m, b = np.polyfit(yt, yp, 1)
                x_fit = np.array([min_val, max_val])
                y_fit = fit_m * x_fit + b
                ax.plot(x_fit, y_fit, color='navy', linestyle='--', lw=1.5, label="Best Fit")

            ax.set_title(f"{title}\n$R^2 = {r2:.2f}$ | MAE $= {mae:.2f}$ cm", fontsize=22)
            
            row_idx = i // cols
            col_idx = i % cols
            
            if col_idx == 0:
                ax.set_ylabel("Predicted (cm)", fontsize=22)
            if row_idx == rows - 1 and col_idx == 1:
                ax.set_xlabel("Ground Truth (cm)", fontsize=22)
            ax.tick_params(axis='both', which='major', labelsize=22)
            ax.grid(True, alpha=0.3, linestyle="--")

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=22)
        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved parity plot to {output_path}")
