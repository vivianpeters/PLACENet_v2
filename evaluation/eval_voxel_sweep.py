#!/usr/bin/env python3
"""
Parameter sweep for voxel postprocessing:
Sweeps probability thresholds, relative thresholds (k_threshold), min cluster volumes, 
and top-K cluster caps to find the optimal operating point.
"""

import argparse
import glob
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

from evaluation.eval_voxel_metrics import (
    extract_cluster_centroids,
    count_voxel_clusters,
    compute_voxel_iou,
    match_voxel_clusters,
    compute_voxel_center_metrics,
)


def run_voxel_sweep(
    labels: np.ndarray,
    pred: np.ndarray,
    threshold_list: List[float] = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    k_threshold_list: List[float] = [0.0, 0.2, 0.35, 0.5],
    min_voxels_list: List[int] = [1, 2],
    top_k_list: List[Optional[int]] = [3, None],
    voxel_size_mm: tuple = (22.9, 22.9, 31.95),
) -> pd.DataFrame:
    """
    Evaluates combinations of voxel postprocessing parameters on a test set.
    """
    batch_size = labels.shape[0]
    
    # Ground truth cluster counts & centroids (fixed at standard 0.5 threshold)
    print("Pre-extracting ground truth clusters...")
    true_counts = np.array([count_voxel_clusters(labels[b], threshold=0.5) for b in range(batch_size)])
    true_centroids_list = [extract_cluster_centroids(labels[b], threshold=0.5) for b in range(batch_size)]
    
    records = []
    total_runs = len(threshold_list) * len(k_threshold_list) * len(min_voxels_list) * len(top_k_list)
    run_idx = 0

    print(f"\nRunning sweep over {total_runs} parameter combinations on {batch_size} samples...")

    for thresh in threshold_list:
        # Precompute Voxel IoU for this base threshold
        iou_vals = compute_voxel_iou(labels, pred, threshold=thresh)
        mean_iou = float(np.mean(iou_vals))

        for k_th in k_threshold_list:
            for min_vox in min_voxels_list:
                for top_k in top_k_list:
                    run_idx += 1
                    
                    # 1. Source counts & predictions
                    pred_counts = np.zeros(batch_size, dtype=int)
                    all_matched_true = []
                    all_matched_pred = []

                    for b in range(batch_size):
                        t_cents = true_centroids_list[b]
                        p_cents = extract_cluster_centroids(
                            pred[b],
                            threshold=thresh,
                            k_threshold=k_th,
                            min_cluster_voxels=min_vox,
                            top_k=top_k,
                        )
                        pred_counts[b] = len(p_cents)

                        # Match clusters
                        matched_pairs, _, _ = match_voxel_clusters(t_cents, p_cents)
                        for p_idx, t_idx in matched_pairs:
                            all_matched_true.append(t_cents[t_idx])
                            all_matched_pred.append(p_cents[p_idx])

                    # 2. Source count classification metrics
                    acc = float(np.mean(pred_counts == true_counts))
                    net_bias = float(np.mean(pred_counts - true_counts))

                    # 3. Positional centroid metrics
                    center_metrics = compute_voxel_center_metrics(
                        all_matched_true, all_matched_pred, voxel_size_mm=voxel_size_mm
                    )
                    
                    if center_metrics is not None:
                        dist_mae_cm = center_metrics["mean_distance_cm"]
                        r2_pos = center_metrics["r2_pos"]
                        matched_n = len(all_matched_true)
                    else:
                        dist_mae_cm = np.nan
                        r2_pos = np.nan
                        matched_n = 0

                    records.append({
                        "threshold": thresh,
                        "k_threshold": k_th,
                        "min_voxels": min_vox,
                        "top_k": "None" if top_k is None else top_k,
                        "count_acc": acc,
                        "net_bias": net_bias,
                        "mean_iou": mean_iou,
                        "dist_mae_cm": dist_mae_cm,
                        "r2_pos": r2_pos,
                        "matched_sources": matched_n,
                    })

    df = pd.DataFrame(records)
    return df


def main():
    parser = argparse.ArgumentParser(description="Sweep voxel postprocessing thresholds and parameters.")
    parser.add_argument("--res-dir", type=str, required=True, help="Directory containing test data and models")
    parser.add_argument("--file-label", type=str, required=True, help="File label (e.g. pu-240_drum_voxel20x20x20_cnn)")
    parser.add_argument("--fold", type=int, default=0, help="Fold to evaluate (default: 0)")
    parser.add_argument("--output-csv", type=str, default="voxel_threshold_sweep.csv", help="Output CSV path")
    args = parser.parse_args()

    res_path = Path(args.res_dir)
    
    # 1. Load test data
    test_pattern = str(res_path / f"*test*{args.file_label}*kf{args.fold}*.npz")
    npz_files = glob.glob(test_pattern)
    if not npz_files:
        test_pattern = str(res_path / f"*data_labels_test*{args.file_label}*.npz")
        npz_files = glob.glob(test_pattern)
    if not npz_files:
        raise FileNotFoundError(f"Could not find test npz file matching '{test_pattern}'")

    print(f"Loading test data from: {npz_files[0]}")
    data = np.load(npz_files[0])
    labels_test = data["labels"]
    data_test = data["data"]

    # 2. Load model & run inference
    model_pattern = str(res_path / f"*{args.file_label}*kf{args.fold}*.keras")
    model_files = glob.glob(model_pattern)
    if not model_files:
        model_pattern = str(res_path / f"*{args.file_label}*kf{args.fold}*.h5")
        model_files = glob.glob(model_pattern)
    if not model_files:
        raise FileNotFoundError(f"Could not find model file matching '{model_pattern}'")

    print(f"Loading model from: {model_files[0]}")
    import tensorflow as tf
    model = tf.keras.models.load_model(model_files[0], compile=False)
    print("Predicting test data...")
    pred = model.predict(data_test, verbose=1)

    # 3. Run sweep
    df = run_voxel_sweep(labels_test, pred)
    
    # Sort by count accuracy descending, then IoU descending
    df_sorted = df.sort_values(by=["count_acc", "mean_iou"], ascending=[False, False])
    
    out_csv = res_path / args.output_csv
    df_sorted.to_csv(out_csv, index=False)
    print(f"\nSaved sweep results to: {out_csv}")
    
    print("\nTop 10 Configurations (ranked by Count Accuracy & IoU):")
    print(df_sorted.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
