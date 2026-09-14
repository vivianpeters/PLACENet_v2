"""Plotting utilities for PLACENet."""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from auxiliaries.geometry import PLACEGeometry, add_drum_surfaces_to_axes
from evaluation.evaluator import extract_boxes_from_model_output, _extract_centres_from_boxes, align_predictions_to_ground_truth
from evaluation.load_model import robust_load_model
from preprocessing.load_data import label_output_dim, label_target_dim, infer_label_mode

import glob
import re
import tensorflow as tf
#from sklearn.metrics import r2_score, roc_curve, precision_recall_curve, roc_auc_score, precision_score, recall_score, f1_score
from pathlib import Path


class PLACENetPlot:
    """Plotting utilities for PLACENet."""

    def __init__(self, drum=None, binx=10, biny=10, binz=10):
        self.drum = drum or PLACEGeometry()
        self.binx = binx
        self.biny = biny
        self.binz = binz

    def plot_voxel_predictions(
        self,
        res_dir,
        file_label: str,
        model_suffix: str = "",
        sample_idx: int = 0,
        threshold: float = 0.5,
    ):
        """
        Plot predicted vs actual voxel grids directly for label_mode='voxel'.
        """
        import glob
        import os
        import re
        import tensorflow as tf
        from sklearn.metrics import r2_score
        from pathlib import Path

        if isinstance(res_dir, str):
            res_dir = Path(res_dir)

        # Load test data
        npz_candidates = sorted(
            glob.glob(str(res_dir / f"data_labels_test_{file_label}_kf*.npz")),
            key=os.path.getmtime,
            reverse=True,
        )

        if not npz_candidates:
            raise FileNotFoundError(f"No test data files found in {res_dir}")

        npz_path = npz_candidates[0]
        m = re.search(r"_kf(\d+)\.npz$", npz_path)
        kfold_str = m.group(1) if m else "0"

        with np.load(npz_path) as npz_file:
            data_test = npz_file["data"]
            labels_test = npz_file["labels"]

        if sample_idx >= len(data_test):
            raise ValueError(f"sample_idx {sample_idx} exceeds test set size {len(data_test)}")

        # Load model
        model_path = res_dir / f"model_{file_label}{model_suffix}_kf{kfold_str}.keras"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        model = robust_load_model(model_path, res_dir, labels_test, data_test)

        # Make predictions
        pred = model.predict(data_test[sample_idx : sample_idx + 1], verbose=0)
        
        y_true = labels_test[sample_idx]
        y_pred = pred[0]

        if y_true.ndim == 4 and y_true.shape[-1] == 1:
            y_true = y_true[..., 0]
        if y_pred.ndim == 4 and y_pred.shape[-1] == 1:
            y_pred = y_pred[..., 0]

        prediction_hist = (y_pred >= threshold).astype(bool)
        actual_hist = (y_true >= 0.5).astype(bool)

        nx, ny, nz = prediction_hist.shape

        # Use the drum's physical properties to create a coordinate grid
        r = self.drum.inner_radius_vox
        cx, cy = self.drum.centre_x_vox, self.drum.centre_y_vox
        z_min, z_max = self.drum.inner_z_min_vox, self.drum.inner_z_max_vox

        # Voxels map directly to coarse grid within inner drum bounds
        # Transform edges to be centered at 0,0 in cm
        x_edges = np.linspace(-r, r, nx + 1)
        y_edges = np.linspace(-r, r, ny + 1)
        z_edges = np.linspace(z_min, z_max, nz + 1)
        
        # create grid for matplotlib voxels
        X, Y, Z = np.meshgrid(x_edges, y_edges, z_edges, indexing='ij')

        fig = plt.figure(figsize=(11, 9), layout="constrained")
        ax1 = fig.add_subplot(111, projection="3d")

        # Manually plot centered drum surfaces
        from auxiliaries.geometry import make_drum_cylinder_surfaces
        inner, outer = make_drum_cylinder_surfaces(self.drum, centre_x=0.0, centre_y=0.0)
        alpha = 0.2
        ax1.plot_surface(*inner, color="gold", alpha=alpha, linewidth=0, antialiased=True)
        ax1.plot_surface(*outer, color="goldenrod", alpha=alpha * 0.85, linewidth=0, antialiased=True)
        
        # Add drum top/bottom rim circles (centered at 0,0)
        theta = np.linspace(0, 2 * np.pi, 40)
        r_outer = self.drum.outer_radius_vox
        ax1.plot(r_outer * np.cos(theta), r_outer * np.sin(theta), self.drum.outer_z_max_vox, color="darkgoldenrod", lw=1.2, ls="--")
        ax1.plot(r_outer * np.cos(theta), r_outer * np.sin(theta), self.drum.outer_z_min_vox, color="darkgoldenrod", lw=1.2, ls="--")
        
        # Plot
        ax1.voxels(X, Y, Z, prediction_hist, facecolor="darkred", edgecolor="darkred", linewidth=0.5, alpha=0.3)
        ax1.voxels(X, Y, Z, actual_hist, facecolor="forestgreen", edgecolor="forestgreen", linewidth=0.5, alpha=0.6)

        legend_elements = [
            Patch(facecolor="forestgreen", edgecolor="forestgreen", label="Ground truth voxels"),
            Patch(facecolor="darkred", edgecolor="darkred", label="Predicted voxels"),
        ]
        ax1.legend(
            handles=legend_elements,
            loc="upper right",
            #bbox_to_anchor=(0.5, -0.05),
            #ncol=2,
            fontsize=20,
        )

        r_limit = r_outer + 2.0
        z_max_val = self.drum.outer_z_max_vox + 2.0
        
        ax1.set_xlim([-r_limit, r_limit])
        ax1.set_ylim([-r_limit, r_limit])
        ax1.set_zlim([self.drum.outer_z_min_vox, z_max_val])
        
        # Keep aspect ratio physically accurate
        ax1.set_box_aspect([2 * r_limit, 2 * r_limit, z_max_val - self.drum.outer_z_min_vox])
        
        ax1.dist = 12.5
        ax1.tick_params(axis='both', which='major', labelsize=20, pad=8)
        ax1.set_xlabel("X (cm)", labelpad=25, fontsize=20)
        ax1.set_ylabel("Y (cm)", labelpad=25, fontsize=20)
        ax1.set_zlabel("Z (cm)", labelpad=30, fontsize=20)

        plt.tight_layout()
        
        output_path = res_dir / f"voxel_predictions_sample{sample_idx}.pdf"
        plt.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0.6)
        print(f"Saved 3D voxel plot to {output_path}")
        return fig

    def plot_3d_bbox_predictions_sample(
        self,
        y_true_sample: np.ndarray,
        y_pred_sample: np.ndarray,
        detector_coords_csv: Optional[Path] = None,
        output_path: Optional[Path] = None,
        sample_id: int = 0,
        unit: str = "cm",
        title: Optional[str] = None,
        show_detectors: bool = False,
        pad_value: float = 0.0,
        label_mode: str = "bbox",
    ) -> plt.Figure:
        """
        Render 3D visualization comparing Ground Truth vs Predicted 3D Bounding Boxes
        within Shielded Drum Geometry, matching PLACENet_GNN visualization style.

        Args:
            y_true_sample: Ground truth boxes of shape (max_sources, 6) or (1, max_sources, 6)
            y_pred_sample: Predicted boxes of shape (max_sources, 6) or (1, max_sources, 6)
            detector_coords_csv: Optional path to CSV containing detector coordinates
            output_path: Optional path to save figure
            sample_id: Sample index for titling and saving
            unit: 'mm' or 'cm' for plot coordinate scale
            title: Custom title string
            show_detectors: Whether to overlay detector positions (default: False)
            pad_value: Padding value for empty box slots (default: -1.0)
        """
        fig = plt.figure(figsize=(11, 9))
        ax = fig.add_subplot(111, projection="3d")

        # 1. Drum Geometry Surfaces
        n_theta, n_z = 40, 20
        theta = np.linspace(0, 2 * np.pi, n_theta)

        if unit == "cm":
            r_outer = self.drum.outer_radius_mm / 10.0  # 28.75 cm
            z_o_max = self.drum.outer_height_mm / 10.0   # 86.5 cm
            z_o_min = 0.0
            r_inner = self.drum.inner_radius_mm / 10.0  # 22.9 cm
            z_i_min = (self.drum.inner_z_center_mm - 0.5 * self.drum.inner_height_mm) / 10.0  # 11.2 cm
            z_i_max = (self.drum.inner_z_center_mm + 0.5 * self.drum.inner_height_mm) / 10.0  # 75.1 cm
        else:
            r_outer = self.drum.outer_radius_mm  # 287.5 mm
            z_o_max = self.drum.outer_height_mm   # 865.0 mm
            z_o_min = 0.0
            r_inner = self.drum.inner_radius_mm  # 229.0 mm
            z_i_min = self.drum.inner_z_center_mm - 0.5 * self.drum.inner_height_mm  # 112.0 mm
            z_i_max = self.drum.inner_z_center_mm + 0.5 * self.drum.inner_height_mm  # 751.0 mm

        # Outer drum cylinder
        z_outer = np.linspace(z_o_min, z_o_max, n_z)
        T_o, Z_o = np.meshgrid(theta, z_outer)
        X_o = r_outer * np.cos(T_o)
        Y_o = r_outer * np.sin(T_o)
        ax.plot_surface(X_o, Y_o, Z_o, color="goldenrod", alpha=0.10, linewidth=0, antialiased=True)

        # Inner drum cylinder
        z_inner = np.linspace(z_i_min, z_i_max, n_z)
        T_i, Z_i = np.meshgrid(theta, z_inner)
        X_i = r_inner * np.cos(T_i)
        Y_i = r_inner * np.sin(T_i)
        ax.plot_surface(X_i, Y_i, Z_i, color="gold", alpha=0.15, linewidth=0, antialiased=True)

        # Drum top/bottom rim circles
        ax.plot(r_outer * np.cos(theta), r_outer * np.sin(theta), z_o_max, color="darkgoldenrod", lw=1.2, ls="--")
        ax.plot(r_outer * np.cos(theta), r_outer * np.sin(theta), z_o_min, color="darkgoldenrod", lw=1.2, ls="--")

        # 2. Detector Positions
        if show_detectors:
            coords = None
            if detector_coords_csv is not None and Path(detector_coords_csv).exists():
                try:
                    import pandas as pd
                    coords_df = pd.read_csv(detector_coords_csv).drop_duplicates(subset=["detectorID"]).sort_values("detectorID")
                    coords = coords_df[["x_mm", "y_mm", "z_mm"]].values
                except Exception:
                    coords = DEFAULT_DETECTOR_COORDS.copy()
            else:
                coords = DEFAULT_DETECTOR_COORDS.copy()

            if coords is not None:
                coords_plot = coords / 10.0 if unit == "cm" else coords
                ax.scatter(coords_plot[:, 0], coords_plot[:, 1], coords_plot[:, 2], color="black", s=90, depthshade=False, label="Detectors (8 Modules)")
                for i in range(len(coords_plot)):
                    ax.text(coords_plot[i, 0] * 1.08, coords_plot[i, 1] * 1.08, coords_plot[i, 2], f"D{i}", fontsize=9, fontweight="bold", color="darkred")

        def draw_3d_wireframe_box(ax, box, color, linestyle="-", label=None):
            if label_mode == "position":
                if len(box) >= 3 and np.all(box[:3] == pad_value): return
                xc, yc, zc = box[:3]
                if xc > 10.0 or zc < 100.0:
                    xc_mm = -500.0 + xc * 10.0; yc_mm = -500.0 + yc * 10.0; zc_mm = zc * 10.0
                else:
                    xc_mm = xc; yc_mm = yc; zc_mm = zc
                xc_p = xc_mm / 10.0 if unit == "cm" else xc_mm
                yc_p = yc_mm / 10.0 if unit == "cm" else yc_mm
                zc_p = zc_mm / 10.0 if unit == "cm" else zc_mm
                ax.scatter([xc_p], [yc_p], [zc_p], color=color, s=100, marker="o", label=label if label else "")
                return

            xw, xc, yw, yc, zw, zc = box[:6]
            if np.all(box[:6] == pad_value):
                return

            # Convert label (voxel cm format vs direct mm format)
            if xc > 10.0 or zc < 100.0:  # voxel cm format
                xw_mm, yw_mm, zw_mm = xw * 10.0, yw * 10.0, zw * 10.0
                xc_mm = -500.0 + xc * 10.0
                yc_mm = -500.0 + yc * 10.0
                zc_mm = zc * 10.0
            else:  # direct mm format
                xw_mm, xc_mm, yw_mm, yc_mm, zw_mm, zc_mm = xw, xc, yw, yc, zw, zc

            if unit == "cm":
                xw_p, xc_p, yw_p, yc_p, zw_p, zc_p = (
                    xw_mm / 10.0, xc_mm / 10.0, yw_mm / 10.0, yc_mm / 10.0, zw_mm / 10.0, zc_mm / 10.0
                )
            else:
                xw_p, xc_p, yw_p, yc_p, zw_p, zc_p = xw_mm, xc_mm, yw_mm, yc_mm, zw_mm, zc_mm

            x_min, x_max = xc_p - xw_p / 2.0, xc_p + xw_p / 2.0
            y_min, y_max = yc_p - yw_p / 2.0, yc_p + yw_p / 2.0
            z_min, z_max = zc_p - zw_p / 2.0, zc_p + zw_p / 2.0

            corners = np.array([
                [x_min, y_min, z_min], [x_max, y_min, z_min], [x_max, y_max, z_min], [x_min, y_max, z_min],
                [x_min, y_min, z_max], [x_max, y_min, z_max], [x_max, y_max, z_max], [x_min, y_max, z_max],
            ])

            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),
                (4, 5), (5, 6), (6, 7), (7, 4),
                (0, 4), (1, 5), (2, 6), (3, 7)
            ]

            for i, (e1, e2) in enumerate(edges):
                ax.plot(
                    [corners[e1, 0], corners[e2, 0]],
                    [corners[e1, 1], corners[e2, 1]],
                    [corners[e1, 2], corners[e2, 2]],
                    color=color, linestyle=linestyle, linewidth=2.0,
                    label=label if (i == 0 and label) else ""
                )

            ax.scatter([xc_p], [yc_p], [zc_p], color=color, s=60, marker="o")

        yt_arr = np.asarray(y_true_sample)
        yp_arr = np.asarray(y_pred_sample)

        if yt_arr.ndim == 3 and yt_arr.shape[0] == 1:
            yt_arr = yt_arr[0]
        if yp_arr.ndim == 3 and yp_arr.shape[0] == 1:
            yp_arr = yp_arr[0]

        # Plot GT boxes (forestgreen)
        gt_count = 0
        target_len = 3 if label_mode == "position" else 6
        for j in range(yt_arr.shape[0]):
            if not np.all(yt_arr[j, :target_len] == pad_value):
                gt_count += 1
                draw_3d_wireframe_box(
                    ax, yt_arr[j], color="forestgreen", linestyle="-",
                    label=("Ground truth positions" if label_mode == "position" else "Ground truth boxes") if gt_count == 1 else None
                )

        # Plot Pred boxes (crimson)
        pred_count = 0
        for j in range(yp_arr.shape[0]):
            if not np.all(yp_arr[j, :target_len] == pad_value):
                pred_count += 1
                draw_3d_wireframe_box(
                    ax, yp_arr[j], color="crimson", linestyle="--",
                    label=("Predicted positions" if label_mode == "position" else "Predicted boxes") if pred_count == 1 else None
                )

        unit_str = "cm" if unit == "cm" else "mm"
        ax.set_xlabel(f"X ({unit_str})", labelpad=25, fontsize=20)
        ax.set_ylabel(f"Y ({unit_str})", labelpad=25, fontsize=20)
        ax.set_zlabel(f"Z ({unit_str})", labelpad=30, fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=20, pad=8)
        ax.dist = 12.5

        if unit == "cm":
            r_limit = (self.drum.outer_radius_mm / 10.0) + 2.0
            z_max = (self.drum.outer_height_mm / 10.0) + 2.0
        else:
            r_limit = self.drum.outer_radius_mm + 20.0
            z_max = self.drum.outer_height_mm + 20.0
            
        ax.set_xlim(-r_limit, r_limit)
        ax.set_ylim(-r_limit, r_limit)
        ax.set_zlim(0, z_max)
        ax.set_box_aspect([2 * r_limit, 2 * r_limit, z_max])

        if title is None:
            title = f"Sample #{sample_id}: 3D Bounding Boxes within Shielded Drum Geometry"
        #ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
        ax.legend(loc="upper right", fontsize=20)

        plt.tight_layout()

        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0.6)
            print(f"Saved 3D wireframe bounding box plot to {output_path}")

        return fig

    def compare_histo(
        self,
        res_dir: Path,
        file_label: str,
        model_suffix: str = "",
        sample_idx: int = 0,
        binx: Optional[int] = None,
        biny: Optional[int] = None,
        binz: Optional[int] = None,
        style: str = "wireframe",
        unit: str = "cm",
        detector_coords_csv: Optional[Path] = None,
        show_detectors: bool = False,
        label_mode: Optional[str] = None,
    ):
        """
        Compare predicted and actual bounding boxes using wireframe drum visualization (PLACENet_GNN style)
        or voxel histogram format.

        Args:
            res_dir: Directory containing test data and model files
            file_label: Label used in file names (e.g., "smooth_l1_jv")
            model_suffix: Optional suffix for model file (e.g., "_ciou")
            sample_idx: Index of sample to visualize (default: 0)
            binx, biny, binz: Histogram bin dimensions (used when style="voxel")
            style: "wireframe" (PLACENet_GNN style 3D drum wireframe) or "voxel" (voxel occupancy grids)
            unit: "mm" or "cm" coordinate scale for wireframe plot
            detector_coords_csv: Path to detector coordinates CSV file
            show_detectors: Whether to overlay detector locations (default: False)
        """
        binx = binx or self.binx
        biny = biny or self.biny
        binz = binz or self.binz

        # Load test data
        npz_candidates = sorted(
            glob.glob(str(res_dir / f"data_labels_test_{file_label}_kf*.npz")),
            key=os.path.getmtime,
            reverse=True,
        )

        if not npz_candidates:
            raise FileNotFoundError(f"No test data files found in {res_dir}")

        model_path = None
        for npz_path in npz_candidates:
            m = re.search(r"_kf(\d+)\.npz$", npz_path)
            kfold_str = m.group(1) if m else "0"
            
            candidate_model_path = res_dir / f"model_{file_label}{model_suffix}_kf{kfold_str}.keras"
            if candidate_model_path.exists():
                model_path = candidate_model_path
                break
                
        if model_path is None:
            raise FileNotFoundError(f"No trained model found matching any data files in {res_dir}")
            
        with np.load(npz_path) as npz_file:
            data_test = npz_file["data"]
            labels_test = npz_file["labels"]

        if sample_idx >= len(data_test):
            raise ValueError(f"sample_idx {sample_idx} exceeds test set size {len(data_test)}")

        model = robust_load_model(model_path, res_dir, labels_test, data_test)

        # Make predictions
        pred = model.predict(data_test[sample_idx : sample_idx + 1], verbose=0)

        # Get single sample
        y_true = labels_test[sample_idx : sample_idx + 1]  # (1, max_sources, 6)
        y_pred = pred[0:1]  # (1, max_sources, 6)

        # Extract boxes/positions if YOLO-style
        actual_label_mode = label_mode or infer_label_mode(last_dim=y_true.shape[-1])
        output_dim = 7 if actual_label_mode == "bbox" else 4
        target_dim = 6 if actual_label_mode == "bbox" else 3

        if y_pred.shape[-1] == output_dim:
            y_pred, _ = extract_boxes_from_model_output(y_pred, confidence_threshold=0.5, label_mode=actual_label_mode)
        if y_true.shape[-1] == output_dim:
            y_true = y_true[..., :target_dim]


        # Apply JV matching
        y_pred_matched = align_predictions_to_ground_truth(y_true, y_pred)

        output_path = res_dir / f"comparison_sample{sample_idx}.pdf"

        if style == "wireframe":
            fig = self.plot_3d_bbox_predictions_sample(
                y_true_sample=y_true[0],
                y_pred_sample=y_pred_matched[0],
                detector_coords_csv=detector_coords_csv,
                output_path=output_path,
                sample_id=sample_idx,
                unit=unit,
                show_detectors=show_detectors,
                label_mode=label_mode or infer_label_mode(last_dim=y_true.shape[-1]),
            )
            return fig

        # Convert box format to voxel histograms (legacy voxel mode)
        prediction_hist = np.zeros((binx, biny, binz))
        actual_hist = np.zeros((binx, biny, binz))

        # Process predicted sources
        for source_idx in range(y_pred_matched.shape[1]):
            source_bbox = y_pred_matched[0, source_idx]

            # Skip padded sources (all -1)
            if np.all(source_bbox == pad_value):
                continue

            # Format: [xwidth, xcentre, ywidth, ycentre, zwidth, zcentre]
            # Box extends from centre ± width/2
            xwidth, xcentre, ywidth, ycentre, zwidth, zcentre = source_bbox
            xmin = xcentre - xwidth / 2.0
            xmax = xcentre + xwidth / 2.0
            ymin = ycentre - ywidth / 2.0
            ymax = ycentre + ywidth / 2.0
            zmin = zcentre - zwidth / 2.0
            zmax = zcentre + zwidth / 2.0

            # Map box boundaries to voxel indices
            x_indices = np.arange(int(xmin), int(xmax) + 1)
            y_indices = np.arange(int(ymin), int(ymax) + 1)
            z_indices = np.arange(int(zmin), int(zmax) + 1)

            # Fill the voxels within the bounding box
            for i in x_indices:
                for j in y_indices:
                    for k in z_indices:
                        if 0 <= i < binx and 0 <= j < biny and 0 <= k < binz:
                            prediction_hist[i, j, k] = 1

        # Process actual sources
        for source_idx in range(y_true.shape[1]):
            source_bbox = y_true[0, source_idx]

            # Skip padded sources (all -1)
            if np.all(source_bbox == pad_value):
                continue

            # Format: [xwidth, xcentre, ywidth, ycentre, zwidth, zcentre]
            # Box extends from centre ± width/2
            xwidth, xcentre, ywidth, ycentre, zwidth, zcentre = source_bbox
            xmin = xcentre - xwidth / 2.0
            xmax = xcentre + xwidth / 2.0
            ymin = ycentre - ywidth / 2.0
            ymax = ycentre + ywidth / 2.0
            zmin = zcentre - zwidth / 2.0
            zmax = zcentre + zwidth / 2.0

            # Map box boundaries to voxel indices
            x_indices = np.arange(int(xmin), int(xmax) + 1)
            y_indices = np.arange(int(ymin), int(ymax) + 1)
            z_indices = np.arange(int(zmin), int(zmax) + 1)

            # Fill the voxels within the bounding box
            for i in x_indices:
                for j in y_indices:
                    for k in z_indices:
                        if 0 <= i < binx and 0 <= j < biny and 0 <= k < binz:
                            actual_hist[i, j, k] = 1

        # Drum cylinder overlays (physical mm → 10 mm voxels, axis at ~50).
        base_width = 6
        base_height = 5
        z_scale_factor = binz / max(binx, biny)
        fig_height = base_height * max(1.0, z_scale_factor * 0.5)

        fig = plt.figure(figsize=(base_width, fig_height), layout="constrained")
        ax1 = fig.add_subplot(111, projection="3d")

        add_drum_surfaces_to_axes(ax1, self.drum)
        # Plot voxels
        ax1.voxels(prediction_hist, facecolor="b", alpha=0.5)
        ax1.voxels(actual_hist, facecolor="r", alpha=1)

        legend_elements = [
            Patch(facecolor="r", edgecolor="r", label="Actual boxes"),
            Patch(facecolor="b", edgecolor="b", label="Predicted boxes"),
        ]
        ax1.legend(
            handles=legend_elements,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.1),
            ncol=2,
            fontsize=14,
        )

        x_lo, x_hi = self.drum.plot_xy_limits()
        z_lo, z_hi = self.drum.plot_z_limits()
        ax1.set_xlim([x_lo, x_hi])
        ax1.set_ylim([x_lo, x_hi])
        z_max_val = min(z_hi, float(binz))
        ax1.set_zlim([z_lo, z_max_val])

        ax1.set_box_aspect([x_hi - x_lo, x_hi - x_lo, z_max_val - z_lo])

        ax1.tick_params(labelsize=14)
        ax1.set_xlabel("x (cm)", fontsize=14)
        ax1.set_ylabel("y (cm)", fontsize=14)
        ax1.set_zlabel("z (cm)", fontsize=14)

        plt.savefig(output_path)
        print(f"Saved comparison plot to {output_path}")
        return fig

    def plot_tdd_sample(res_dir, sample_idx=0, detector_idx=0):
        res_path = Path(res_dir)
        npz_candidates = list(res_path.glob("*_kf*.npz"))
        if not npz_candidates:
            print(f"No .npz files found in {res_dir}")
            return
        
        npz_path = npz_candidates[0]
        with np.load(npz_path) as npz_file:
            data = npz_file["data"]
            
        if sample_idx >= len(data):
            print(f"Sample index {sample_idx} is out of bounds for data of size {len(data)}")
            return
            
        sample = data[sample_idx]
        if sample.ndim == 3 and sample.shape[-1] == 1:
            tdd = sample[..., 0]
        elif sample.ndim == 2:
            tdd = sample
        else:
            print(f"Unexpected shape: {sample.shape}")
            return
            
        n_detectors, n_bins = tdd.shape
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

        axes[0].plot(np.arange(n_bins), tdd[detector_idx], lw=1.5, color="indigo")
        axes[0].set_title(f"Detector pair {detector_idx}", fontsize=18)
        axes[0].set_xlabel("Time (ns)", fontsize=20)
        axes[0].set_ylabel("Counts (normalized)", fontsize=20)
        axes[0].grid(alpha=0.3)
        axes[0].tick_params(labelsize=20)

        im = axes[1].imshow(tdd, aspect="auto", origin="lower", cmap="viridis")
        axes[1].set_title(f"Complete TOF Array", fontsize=18)
        axes[1].set_xlabel("Time (ns)", fontsize=20)
        axes[1].set_ylabel("Detector pair index", fontsize=20)
        axes[1].tick_params(labelsize=20)
        cbar = fig.colorbar(im, ax=axes[1])
        cbar.set_label("Counts (normalized)", fontsize=20)
        cbar.ax.tick_params(labelsize=20)

        out_path = res_path / f"tdd_sample{sample_idx}.pdf"
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        print(f"Saved TDD plot to {out_path}")