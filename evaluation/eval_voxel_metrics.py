import numpy as np
import scipy.ndimage as ndimage
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import r2_score

def extract_cluster_centroids(voxel_grid: np.ndarray, threshold: float = 0.5):
    """
    Extract centers of mass for connected components in a 3D voxel grid.
    
    Returns:
        list of np.ndarray (cx, cy, cz) centroid coordinates in voxel units.
    """
    binary_grid = (voxel_grid >= threshold).astype(np.uint8)
    
    # Strip trailing dimension if it is shape (20, 20, 20, 1)
    if binary_grid.ndim == 4 and binary_grid.shape[-1] == 1:
        binary_grid = binary_grid[..., 0]
    
    val_grid = voxel_grid
    if val_grid.ndim == 4 and val_grid.shape[-1] == 1:
        val_grid = val_grid[..., 0]
        
    s = ndimage.generate_binary_structure(3, 3)
    labeled_grid, num_features = ndimage.label(binary_grid, structure=s)
    
    if num_features == 0:
        return []
        
    centroids = ndimage.center_of_mass(val_grid, labeled_grid, range(1, num_features + 1))
    return [np.array(c) for c in centroids]

def count_voxel_clusters(voxel_grid: np.ndarray, threshold: float = 0.5):
    """
    Count the number of disconnected clusters in a voxel grid.
    """
    binary_grid = (voxel_grid >= threshold).astype(np.uint8)
    if binary_grid.ndim == 4 and binary_grid.shape[-1] == 1:
        binary_grid = binary_grid[..., 0]
        
    s = ndimage.generate_binary_structure(3, 3)
    _, num_features = ndimage.label(binary_grid, structure=s)
    return num_features

def compute_voxel_iou(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5):
    """
    Compute Voxel IoU for a batch of predictions.
    """
    true_bin = (y_true >= 0.5).astype(np.float32)
    pred_bin = (y_pred >= threshold).astype(np.float32)
    
    if true_bin.ndim == 5 and true_bin.shape[-1] == 1:
        true_bin = true_bin[..., 0]
    if pred_bin.ndim == 5 and pred_bin.shape[-1] == 1:
        pred_bin = pred_bin[..., 0]
        
    inter = np.sum(pred_bin * true_bin, axis=(1, 2, 3))
    union = np.sum((pred_bin + true_bin > 0).astype(np.float32), axis=(1, 2, 3))
    iou = inter / (union + 1e-7)
    return iou

def match_voxel_clusters(true_centroids: list, pred_centroids: list):
    """
    Match predicted centroids to ground truth centroids using Euclidean distance.
    Returns:
        matched_pairs: list of tuples (pred_idx, true_idx)
        unmatched_pred: list of pred_idx
        unmatched_true: list of true_idx
    """
    n_true = len(true_centroids)
    n_pred = len(pred_centroids)
    
    if n_true == 0 or n_pred == 0:
        return [], list(range(n_pred)), list(range(n_true))
        
    cost_matrix = np.zeros((n_pred, n_true))
    for i in range(n_pred):
        for j in range(n_true):
            cost_matrix[i, j] = np.linalg.norm(pred_centroids[i] - true_centroids[j])
            
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    matched_pairs = list(zip(row_ind, col_ind))
    
    unmatched_pred = [i for i in range(n_pred) if i not in row_ind]
    unmatched_true = [j for j in range(n_true) if j not in col_ind]
    
    return matched_pairs, unmatched_pred, unmatched_true

def compute_voxel_center_metrics(true_centroids_matched, pred_centroids_matched, voxel_size_mm=(22.9, 22.9, 31.95)):
    """
    Compute Center Distance MAE and Positional R2 for matched clusters.
    """
    if len(true_centroids_matched) == 0:
        return None
        
    scale_mm = np.array(voxel_size_mm)
    
    yt_mm = np.array(true_centroids_matched) * scale_mm
    yp_mm = np.array(pred_centroids_matched) * scale_mm
    
    # 3D Euclidean distance in mm
    distances = np.linalg.norm(yt_mm - yp_mm, axis=1)
    mean_distance = np.mean(distances)
    
    # Per-axis MAE in mm
    mae_x = np.mean(np.abs(yt_mm[:, 0] - yp_mm[:, 0]))
    mae_y = np.mean(np.abs(yt_mm[:, 1] - yp_mm[:, 1]))
    mae_z = np.mean(np.abs(yt_mm[:, 2] - yp_mm[:, 2]))
    
    # Positional R2 (scale independent)
    r2_pos = r2_score(yt_mm, yp_mm)
    
    # Return coordinates in cm for parity plotting which expects cm
    yt_cm = yt_mm / 10.0
    yp_cm = yp_mm / 10.0
    
    return {
        "mean_distance_mm": mean_distance,
        "mae_x_mm": mae_x,
        "mae_y_mm": mae_y,
        "mae_z_mm": mae_z,
        "r2_pos": r2_pos,
        "matched_true": yt_cm,
        "matched_pred": yp_cm
    }
