"""Evaluation and plotting for PLACENet using a configuration-driven Evaluator."""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    r2_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
)

from preprocessing.load_data import infer_label_mode, label_output_dim, label_target_dim
from evaluation.eval_config import EvaluationConfig, EvalStepConfig
from evaluation.load_model import robust_load_model
from evaluation.eval_metrics import compute_r2_scores, compute_mae_metrics
from evaluation.eval_bbox_confidence import compute_classification_metrics, compute_slot_confidence_metrics, count_valid_sources, empty_slot_accuracy
from plot.plot_parity import plot_pred_vs_gt_parity
from plot.plot_confusion_matrix import plot_voxel_confusion_matrix, plot_combined_confusion_matrix, plot_slot_confusion_matrix
from evaluation.eval_bbox_iou import get_iou_by_source_count
from evaluation.eval_voxel_metrics import extract_cluster_centroids, count_voxel_clusters, compute_voxel_iou, match_voxel_clusters, compute_voxel_center_metrics

# Box layout: [xwidth, xcentre, ywidth, ycentre, zwidth, zcentre]
_CENTRE_IDX = (1, 3, 5)

def _extract_centres_from_boxes(boxes: np.ndarray) -> np.ndarray:
    """(batch, max_sources, 6) -> (batch, max_sources, 3) centre channels."""
    boxes = np.asarray(boxes, dtype=np.float64)
    centres = boxes[..., _CENTRE_IDX]
    pad_mask = np.all(boxes == 0.0, axis=-1)
    if np.any(pad_mask):
        centres = centres.copy()
        centres[pad_mask] = 0.0
    return centres

def _resolve_label_mode(label_mode=None, array=None):
    if label_mode is not None:
        return label_mode
    if array is not None:
        return infer_label_mode(last_dim=array.shape[-1])
    return "bbox"

def extract_targets_from_model_output(
    y_pred,
    label_mode="bbox",
    confidence_threshold=0.5,
):
    """Extract regression targets from YOLO-style output and filter by confidence."""
    label_mode = _resolve_label_mode(label_mode, y_pred)
    target_dim = label_target_dim(label_mode)
    output_dim = label_output_dim(label_mode)
    if y_pred.shape[-1] == output_dim:
        targets = y_pred[..., :target_dim].copy()
        confidences = y_pred[..., target_dim]
        low_conf_mask = confidences < confidence_threshold
        targets[low_conf_mask] = 0.0
        return targets, confidences
    return y_pred, None

def extract_boxes_from_model_output(y_pred, confidence_threshold=0.5, label_mode="bbox"):
    return extract_targets_from_model_output(
        y_pred,
        label_mode=label_mode,
        confidence_threshold=confidence_threshold,
    )

def _get_valid_boxes(y_true_b: np.ndarray, y_pred_b: np.ndarray):
    """
    Get valid (non-padded) box rows for one batch item.
    """
    true_valid_mask = ~np.all(y_true_b == 0.0, axis=-1)
    pred_valid_mask = ~np.all(y_pred_b == 0.0, axis=-1)
    true_valid = y_true_b[true_valid_mask]
    pred_valid = y_pred_b[pred_valid_mask]
    return true_valid_mask, pred_valid_mask, true_valid, pred_valid

def _compute_box_cost(
    pred_box: np.ndarray,
    true_box: np.ndarray,
    cost_metric: str = "hybrid",
    delta: float = 1.0,
    smooth_weight: float = 1.0,
    ciou_weight: float = 1.0,
) -> float:
    """Compute pair cost between 6D pred_box and true_box."""
    if cost_metric == "l2":
        return float(np.linalg.norm(pred_box[:6] - true_box[:6]))

    if len(pred_box) < 6:
        diff = np.abs(pred_box - true_box[:len(pred_box)])
        smooth_l1_vals = np.where(diff < delta, 0.5 * (diff ** 2), delta * diff - 0.5 * (delta ** 2))
        return float(np.sum(smooth_l1_vals))

    xw_p, xc_p, yw_p, yc_p, zw_p, zc_p = pred_box[:6]
    xw_t, xc_t, yw_t, yc_t, zw_t, zc_t = true_box[:6]

    diff = np.abs(pred_box[:6] - true_box[:6])
    smooth_l1_vals = np.where(diff < delta, 0.5 * (diff ** 2), delta * diff - 0.5 * (delta ** 2))
    smooth_l1_cost = float(np.sum(smooth_l1_vals))
    if cost_metric == "smooth_l1":
        return smooth_l1_cost

    x_min_t, x_max_t = xc_t - abs(xw_t)/2, xc_t + abs(xw_t)/2
    y_min_t, y_max_t = yc_t - abs(yw_t)/2, yc_t + abs(yw_t)/2
    z_min_t, z_max_t = zc_t - abs(zw_t)/2, zc_t + abs(zw_t)/2

    x_min_p, x_max_p = xc_p - abs(xw_p)/2, xc_p + abs(xw_p)/2
    y_min_p, y_max_p = yc_p - abs(yw_p)/2, yc_p + abs(yw_p)/2
    z_min_p, z_max_p = zc_p - abs(zw_p)/2, zc_p + abs(zw_p)/2

    inter = (
        max(min(x_max_t, x_max_p) - max(x_min_t, x_min_p), 0) *
        max(min(y_max_t, y_max_p) - max(y_min_t, y_min_p), 0) *
        max(min(z_max_t, z_max_p) - max(z_min_t, z_min_p), 0)
    )
    vol_t = abs(xw_t) * abs(yw_t) * abs(zw_t)
    vol_p = abs(xw_p) * abs(yw_p) * abs(zw_p)
    union = vol_t + vol_p - inter + 1e-7
    iou = inter / union

    enc_x_min, enc_x_max = min(x_min_t, x_min_p), max(x_max_t, x_max_p)
    enc_y_min, enc_y_max = min(y_min_t, y_min_p), max(y_max_t, y_max_p)
    enc_z_min, enc_z_max = min(z_min_t, z_min_p), max(z_max_t, z_max_p)
    c2 = (enc_x_max - enc_x_min)**2 + (enc_y_max - enc_y_min)**2 + (enc_z_max - enc_z_min)**2 + 1e-7

    rho2 = (xc_p - xc_t)**2 + (yc_p - yc_t)**2 + (zc_p - zc_t)**2
    ciou = iou - (rho2 / c2)
    ciou_cost = float(1.0 - ciou)

    if cost_metric == "ciou":
        return ciou_cost

    return smooth_weight * smooth_l1_cost + ciou_weight * ciou_cost

def _compute_box_assignment(
    true_valid: np.ndarray, pred_valid: np.ndarray, pred_conf_valid: np.ndarray = None,
    cost_metric: str = "hybrid", delta: float = 1.0, smooth_weight: float = 0.1, ciou_weight: float = 1.0,
    confidence_weight: float = 2.0
) -> List[tuple]:
    """Compute assignment of predicted boxes to ground truth boxes using Jonker-Volgenant (JV)."""
    n_true = len(true_valid)
    n_pred = len(pred_valid)
    if n_true == 0 or n_pred == 0:
        return []

    cost_matrix = np.zeros((n_pred, n_true))
    for i in range(n_pred):
        for j in range(n_true):
            cost_matrix[i, j] = _compute_box_cost(
                pred_valid[i], true_valid[j], cost_metric=cost_metric,
                delta=delta, smooth_weight=smooth_weight, ciou_weight=ciou_weight
            )

    if pred_conf_valid is not None:
        cost_scale = np.mean(cost_matrix) + 1.0
        for i in range(n_pred):
            for j in range(n_true):
                cost_matrix[i, j] -= (cost_scale * confidence_weight * pred_conf_valid[i])

    size = max(n_true, n_pred)
    cost_square = np.full((size, size), 1e10)
    cost_square[:n_pred, :n_true] = cost_matrix
    row_ind, col_ind = linear_sum_assignment(cost_square)
    return [(i, j) for i, j in zip(row_ind, col_ind) if i < n_pred and j < n_true]

def align_predictions_to_ground_truth(
    labels: np.ndarray, pred: np.ndarray, pred_conf: np.ndarray = None,
    cost_metric: str = "hybrid", delta: float = 1.0, smooth_weight: float = 0.1, ciou_weight: float = 1.0,
    confidence_weight: float = 2.0
):
    """Align predictions to ground truth slots."""
    batch_size = labels.shape[0]
    max_sources = labels.shape[1]
    target_dim = pred.shape[2]
    pred_aligned = np.zeros_like(pred)
    pred_conf_aligned = np.zeros((batch_size, max_sources)) if pred_conf is not None else None

    for b in range(batch_size):
        true_valid_mask, pred_valid_mask, true_valid, pred_valid = _get_valid_boxes(labels[b], pred[b])
        conf_valid = pred_conf[b][pred_valid_mask] if pred_conf is not None else None
        
        matches = _compute_box_assignment(
            true_valid, pred_valid, pred_conf_valid=conf_valid,
            cost_metric=cost_metric, delta=delta, smooth_weight=smooth_weight,
            ciou_weight=ciou_weight, confidence_weight=confidence_weight
        )
        
        true_indices = np.where(true_valid_mask)[0]
        pred_indices = np.where(pred_valid_mask)[0]
        
        matched_pred_indices = set()
        for p_idx, t_idx in matches:
            original_t_idx = true_indices[t_idx]
            original_p_idx = pred_indices[p_idx]
            pred_aligned[b, original_t_idx] = pred[b, original_p_idx]
            if pred_conf is not None:
                pred_conf_aligned[b, original_t_idx] = pred_conf[b, original_p_idx]
            matched_pred_indices.add(original_p_idx)
            
        unmatched_pred_indices = [p for p in pred_indices if p not in matched_pred_indices]
        if pred_conf is not None:
            # Sort unmatched predictions by confidence (highest first)
            unmatched_pred_indices.sort(key=lambda p: pred_conf[b, p], reverse=True)
            
        empty_true_indices = np.where(~true_valid_mask)[0]
        
        for i, original_p_idx in enumerate(unmatched_pred_indices):
            if i < len(empty_true_indices):
                original_t_idx = empty_true_indices[i]
                pred_aligned[b, original_t_idx] = pred[b, original_p_idx]
                if pred_conf is not None:
                    pred_conf_aligned[b, original_t_idx] = pred_conf[b, original_p_idx]
                
    return pred_aligned, pred_conf_aligned

class Evaluator:
    def __init__(self, config: EvaluationConfig, results_dir: str, data_dir: str):
        self.config = config
        self.res_dir = Path(results_dir)
        self.data_dir = Path(data_dir)
        self.res_dir.mkdir(parents=True, exist_ok=True)
        
    def _is_fold_requested(self, step_config: EvalStepConfig, fold_num: int):
        if not step_config.enabled:
            return False
        if step_config.folds == "all":
            return True
        if step_config.folds == "each":
            return True
        if isinstance(step_config.folds, list):
            return fold_num in step_config.folds
        return False
        
    def _should_run_aggregated(self, step_config: EvalStepConfig):
        if not step_config.enabled:
            return False
        return step_config.folds == "all"
        
    def _should_run_individual(self, step_config: EvalStepConfig, fold_num: int):
        if not step_config.enabled:
            return False
        if step_config.folds == "each":
            return True
        if isinstance(step_config.folds, list):
            return fold_num in step_config.folds
        return False

    def _process_fold(self, fold_num: int):
        kfold_str = str(fold_num)
        val_data_pattern = str(self.res_dir / f"data_labels_test_{self.config.file_label}_kf{kfold_str}.npz")
        val_files = glob.glob(val_data_pattern)
        if not val_files:
            raise FileNotFoundError(f"No validation data found for fold {fold_num}")
            
        data = np.load(val_files[0])
        labels_test = data["labels"]
        data_test = data["data"]
        
        # Determine actual label mode if not provided properly
        if self.config.label_mode not in ("bbox", "position", "voxel"):
            actual_label_mode = "bbox" if labels_test.shape[-1] == 6 else "position"
        else:
            actual_label_mode = self.config.label_mode

        model_pattern = str(self.res_dir / f"*{self.config.file_label}*kf{kfold_str}*.keras")
        model_files = glob.glob(model_pattern)
        if not model_files:
            model_pattern = str(self.res_dir / f"*{self.config.file_label}*kf{kfold_str}*.h5")
            model_files = glob.glob(model_pattern)
            if not model_files:
                raise FileNotFoundError(f"No model found for fold {fold_num}")
                
        model_path = model_files[0]
        
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
        except Exception:
            model = robust_load_model(model_path, self.res_dir, labels_test, data_test)
            
        pred_raw = model.predict(data_test, verbose=0)
        
        if actual_label_mode == "voxel":
            return {
                "labels": labels_test,
                "pred": pred_raw,
                "actual_label_mode": actual_label_mode
            }
            
        pred, pred_confidences = extract_boxes_from_model_output(
            pred_raw, confidence_threshold=0.5, label_mode=actual_label_mode
        )
        
        pred_matched, pred_conf_matched = align_predictions_to_ground_truth(
            labels_test, pred, pred_confidences, cost_metric=self.config.match_mode
        )
        
        pred_unfiltered, _ = extract_boxes_from_model_output(
            pred_raw, confidence_threshold=0.0, label_mode=actual_label_mode
        )
        _, pred_conf_matched_raw = align_predictions_to_ground_truth(
            labels_test, pred_unfiltered, pred_confidences, cost_metric=self.config.match_mode
        )
        
        return {
            "labels": labels_test,
            "pred": pred,
            "pred_conf": pred_confidences,
            "pred_matched": pred_matched,
            "pred_conf_matched": pred_conf_matched,
            "pred_conf_matched_raw": pred_conf_matched_raw,
            "actual_label_mode": actual_label_mode
        }

    def _run_step_metrics(self, data, step_config: EvalStepConfig, suffix: str):
        if data["actual_label_mode"] == "voxel":
            return
            
        labels = data["labels"]
        pred_matched = data["pred_matched"]
        max_sources = labels.shape[1]
        target_dim = pred_matched.shape[2]
        
        if labels.shape[-1] > target_dim:
            labels = labels[..., :target_dim]
        
        if step_config.print_results:
            r2_res = compute_r2_scores(labels, pred_matched, max_sources, target_dim, data["actual_label_mode"])
            mae_res = compute_mae_metrics(labels, pred_matched, max_sources, target_dim)
            
            output_lines = [
                f"\n--- Metrics ({suffix}) ---",
                f"R2 Valid Slots: {r2_res['r2_valid']:.4f}",
                f"R2 All Slots:   {r2_res['r2_all']:.4f}",
                f"MAE:            {mae_res['mae']:.4f} cm"
            ]
            for line in output_lines: print(line)
            
            with open(self.res_dir / f"evaluation_metrics_{self.config.file_label}.txt", "a") as f:
                f.write("\n".join(output_lines) + "\n")
            
        if step_config.plot_results:
            out_path = self.res_dir / f"parity_{self.config.file_label}_{suffix}.pdf"
            plot_pred_vs_gt_parity(labels, pred_matched, output_path=out_path, 
                                 label_mode=data["actual_label_mode"], 
                                 do_plot=True, do_print=False)

    def _run_step_confidence(self, data, step_config: EvalStepConfig, suffix: str):
        if data["actual_label_mode"] == "voxel":
            return
            
        if data["pred_conf_matched"] is None:
            if self.config.verbose: print(f"No confidence output for {suffix}.")
            return
            
        res = compute_slot_confidence_metrics(data["labels"], data["pred_conf_matched_raw"])
        
        if step_config.print_results:
            output_lines = [
                f"\n--- Confidence Stats ({suffix}) ---",
                f"Accuracy:  {res['confidence_accuracy']:.4f}",
                f"Precision: {res['confidence_precision']:.4f}",
                f"Recall:    {res['confidence_recall']:.4f}",
                f"F1:        {res['confidence_f1']:.4f}",
                f"AUC:       {res['confidence_auc']:.4f}"
            ]
            for line in output_lines: print(line)
            
            with open(self.res_dir / f"evaluation_metrics_{self.config.file_label}.txt", "a") as f:
                f.write("\n".join(output_lines) + "\n")
            
        if step_config.plot_results:
            cm = np.array(res["confidence_confusion_matrix"])
            out_path = self.res_dir / f"confidence_cm_{self.config.file_label}_{suffix}.pdf"
            plot_slot_confusion_matrix(cm, out_path)
            
            try:
                from plot.plot_bbox_confidence import plot_confidence_by_slot_type
                true_conf = np.any(data["labels"] != 0.0, axis=-1).astype(float)
                dist_path = self.res_dir / f"confidence_dist_{self.config.file_label}_{suffix}.pdf"
                plot_confidence_by_slot_type(data["pred_conf_matched_raw"], true_conf, output_path=dist_path)
            except Exception as e:
                if self.config.verbose: print(f"Could not plot confidence by slot type: {e}")

    def _run_step_confusion_matrices(self, data, step_config: EvalStepConfig, suffix: str):
        if data["actual_label_mode"] == "voxel":
            labels = data["labels"]
            pred = data["pred"]
            batch_size = labels.shape[0]
            
            true_counts = np.array([count_voxel_clusters(labels[b], self.config.voxel_threshold) for b in range(batch_size)])
            pred_counts = np.array([count_voxel_clusters(pred[b], self.config.voxel_threshold) for b in range(batch_size)])
            
            # For plotting confusion matrix, determine max sources dynamically based on data
            max_sources_dyn = max(np.max(true_counts), np.max(pred_counts))
            all_labels = np.arange(0, max_sources_dyn + 1)
        else:
            labels = data["labels"]
            pred_matched = data["pred_matched"]
            pred_conf = data["pred_conf_matched"]
            
            max_sources = labels.shape[1]
            all_labels = np.arange(0, max_sources + 1)
            
            true_counts = count_valid_sources(labels)
            pred_counts = count_valid_sources(pred_matched, pred_conf)
            
        res = compute_classification_metrics(true_counts, pred_counts, all_labels, "jv")
            
        if step_config.plot_results:
            cm = np.array(res["confusion_matrix"])
            out_path = self.res_dir / f"source_count_cm_{self.config.file_label}_{suffix}.pdf"
            plot_voxel_confusion_matrix(cm, all_labels, out_path)

    def _run_step_performance_by_count(self, data, step_config: EvalStepConfig, suffix: str):
        if data["actual_label_mode"] == "voxel":
            labels = data["labels"]
            pred = data["pred"]
            batch_size = labels.shape[0]
            
            if step_config.plot_results:
                iou_vals = compute_voxel_iou(labels, pred, self.config.voxel_threshold)
                true_counts = np.array([count_voxel_clusters(labels[b], self.config.voxel_threshold) for b in range(batch_size)])
                
                iou_res = [{"num_sources": count, "iou": float(iou)} for count, iou in zip(true_counts, iou_vals)]
                if len(iou_res) > 0:
                    df = pd.DataFrame(iou_res)
                    out_path = self.res_dir / f"iou_vs_complexity_{self.config.file_label}_{suffix}.pdf"
                    
                    import seaborn as sns
                    plt.figure(figsize=(6, 6))
                    sns.boxplot(data=df, x="num_sources", y="iou", palette="Blues")
                    plt.xlabel("Number of true sources", fontsize=24)
                    plt.ylabel("Voxel IoU", fontsize=24)
                    plt.ylim(0, 1.05)
                    plt.grid(True, linestyle="--", alpha=0.3, axis="y")
                    plt.tick_params(labelsize=24)
                    plt.tight_layout()
                    plt.savefig(out_path, dpi=200)
                    plt.close()
                    if self.config.verbose: print(f"Saved performance vs complexity plot to {out_path}")
        else:
            labels = data["labels"]
            pred_matched = data["pred_matched"]
            
            if step_config.plot_results:
                iou_res = get_iou_by_source_count(labels, pred_matched, data["actual_label_mode"])
                if len(iou_res) > 0:
                    df = pd.DataFrame(iou_res)
                    out_path = self.res_dir / f"iou_by_count_{self.config.file_label}_{suffix}.pdf"
                    try:
                        from plot.plot_performance_by_count import plot_iou_vs_source_count
                        plot_iou_vs_source_count(df, output_path=out_path)
                    except ImportError:
                        if self.config.verbose: print("Could not import PLACENetPlot for performance by count plot.")

    def _run_step_voxel_cluster_parity(self, data, step_config: EvalStepConfig, suffix: str):
        if data["actual_label_mode"] != "voxel":
            return
            
        labels = data["labels"]
        pred = data["pred"]
        batch_size = labels.shape[0]
        
        all_matched_true = []
        all_matched_pred = []
        
        for b in range(batch_size):
            t_cents = extract_cluster_centroids(labels[b], threshold=0.5)
            p_cents = extract_cluster_centroids(pred[b], threshold=self.config.voxel_threshold)
            
            matched_pairs, _, _ = match_voxel_clusters(t_cents, p_cents)
            for p_idx, t_idx in matched_pairs:
                all_matched_true.append(t_cents[t_idx])
                all_matched_pred.append(p_cents[p_idx])
                
        metrics = compute_voxel_center_metrics(all_matched_true, all_matched_pred, voxel_size_mm=self.config.voxel_size_mm)
        
        if metrics is None:
            if self.config.verbose: print(f"No matched voxel clusters found for parity in {suffix}.")
            return
            
        if step_config.print_results:
            log_text = (
                f"\n--- Voxel Center Distance Parity Metrics ({suffix}) ---\n"
                f"Mean 3D Distance Error: {metrics['mean_distance_mm']:.4f} mm\n"
                f"MAE X: {metrics['mae_x_mm']:.4f} mm\n"
                f"MAE Y: {metrics['mae_y_mm']:.4f} mm\n"
                f"MAE Z: {metrics['mae_z_mm']:.4f} mm\n"
                f"Positional R2: {metrics['r2_pos']:.4f}\n"
            )
            print(log_text)
            log_path = self.res_dir / f"voxel_center_distance_{self.config.file_label}.txt"
            with open(log_path, "a") as f:
                f.write(log_text)
            
        if step_config.plot_results:
            yt = np.expand_dims(metrics["matched_true"], axis=0)
            yp = np.expand_dims(metrics["matched_pred"], axis=0)
            
            out_path = self.res_dir / f"voxel_center_parity_{self.config.file_label}_{suffix}.pdf"
            plot_pred_vs_gt_parity(yt, yp, output_path=out_path, label_mode="position", 
                                 do_plot=True, do_print=False)

    def run(self):
        metrics_log_path = self.res_dir / f"evaluation_metrics_{self.config.file_label}.txt"
        if metrics_log_path.exists():
            metrics_log_path.unlink()
            
        folds_to_process = set()
        for i in range(0, self.config.n_folds):
            if any([
                self._is_fold_requested(self.config.metrics, i),
                self._is_fold_requested(self.config.confidence, i),
                self._is_fold_requested(self.config.confusion_matrices, i),
                self._is_fold_requested(self.config.performance_by_count, i),
                self._is_fold_requested(self.config.voxel_cluster_parity, i),
            ]):
                folds_to_process.add(i)

        if not folds_to_process:
            if self.config.verbose: print("No evaluation steps requested. Exiting.")
            return

        aggregated_data = {
            "labels": [], "pred": [], "pred_conf": [],
            "pred_matched": [], "pred_conf_matched": [], "pred_conf_matched_raw": [],
            "actual_label_mode": self.config.label_mode
        }

        for kfold in sorted(list(folds_to_process)):
            if self.config.verbose: print(f"\n{'='*40}\nProcessing fold {kfold}\n{'='*40}")
            data = self._process_fold(kfold)
            
            # Accumulate for 'all'
            for key in aggregated_data:
                if key != "actual_label_mode" and key in data and data[key] is not None:
                    aggregated_data[key].append(data[key])
            aggregated_data["actual_label_mode"] = data["actual_label_mode"]

            # Run individual fold steps
            if self._should_run_individual(self.config.metrics, kfold):
                self._run_step_metrics(data, self.config.metrics, f"kf{kfold}")
            if self._should_run_individual(self.config.confidence, kfold):
                self._run_step_confidence(data, self.config.confidence, f"kf{kfold}")
            if self._should_run_individual(self.config.confusion_matrices, kfold):
                self._run_step_confusion_matrices(data, self.config.confusion_matrices, f"kf{kfold}")
            if self._should_run_individual(self.config.performance_by_count, kfold):
                self._run_step_performance_by_count(data, self.config.performance_by_count, f"kf{kfold}")
            if self._should_run_individual(self.config.voxel_cluster_parity, kfold):
                self._run_step_voxel_cluster_parity(data, self.config.voxel_cluster_parity, f"kf{kfold}")

        # Run aggregated steps
        has_aggregated = any([
            self._should_run_aggregated(self.config.metrics),
            self._should_run_aggregated(self.config.confidence),
            self._should_run_aggregated(self.config.confusion_matrices),
            self._should_run_aggregated(self.config.performance_by_count),
            self._should_run_aggregated(self.config.voxel_cluster_parity)
        ])

        if has_aggregated and len(aggregated_data["labels"]) > 0:
            if self.config.verbose: print(f"\n{'='*40}\nProcessing Aggregated (All Folds)\n{'='*40}")
            # Concatenate all lists
            concat_data = {"actual_label_mode": aggregated_data["actual_label_mode"]}
            for key in aggregated_data:
                if key != "actual_label_mode":
                    if len(aggregated_data[key]) > 0:
                        concat_data[key] = np.concatenate(aggregated_data[key], axis=0)
                    else:
                        concat_data[key] = None

            if self._should_run_aggregated(self.config.metrics):
                self._run_step_metrics(concat_data, self.config.metrics, "all_folds")
            if self._should_run_aggregated(self.config.confidence):
                self._run_step_confidence(concat_data, self.config.confidence, "all_folds")
            if self._should_run_aggregated(self.config.confusion_matrices):
                self._run_step_confusion_matrices(concat_data, self.config.confusion_matrices, "all_folds")
            if self._should_run_aggregated(self.config.performance_by_count):
                self._run_step_performance_by_count(concat_data, self.config.performance_by_count, "all_folds")
            if self._should_run_aggregated(self.config.voxel_cluster_parity):
                self._run_step_voxel_cluster_parity(concat_data, self.config.voxel_cluster_parity, "all_folds")

        # Loss curves
        if self.config.loss_curves.enabled:
            if self.config.verbose: 
                print(f"\n{'='*40}\nPlotting Loss Curves\n{'='*40}")
            logs_dir = Path(self.res_dir) / "logs"
            if logs_dir.exists():
                from plot.plot_loss_curves import plot_loss_curves_from_tb
                output_path = Path(self.res_dir) / f"loss_curves_{self.config.file_label}.pdf"
                plot_loss_curves_from_tb(logs_dir, output_path)
            else:
                print("WARNING: No TensorBoard logs found in results dir. Cannot plot loss curves.")
