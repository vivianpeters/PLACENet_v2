import numpy as np

def compute_position_mae_per_slot_stats(y_true, y_pred, label_mode="position"):
    """Mean L2 position error per matched valid slot."""
    if label_mode not in ("position", "bbox"):
        return np.nan, np.nan, 0

    if label_mode == "bbox":
        y_true = _extract_centres_from_boxes(y_true)
        y_pred = _extract_centres_from_boxes(y_pred)

    target_dim = label_target_dim("position")
    output_dim = label_output_dim("position")
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape[-1] == output_dim:
        y_true = y_true[..., :target_dim]
    if y_pred.shape[-1] == output_dim:
        y_pred = y_pred[..., :target_dim]

    valid = np.all(y_true != 0.0, axis=-1) & np.all(y_pred != 0.0, axis=-1)
    if not np.any(valid):
        return np.nan, np.nan, 0

    diff = np.linalg.norm(y_true - y_pred, axis=-1)
    values = diff[valid]
    count = len(values)
    mean_mae = float(np.mean(values))
    std_mae = float(np.std(values, ddof=1)) if count > 1 else 0.0
    return mean_mae, std_mae, count
from sklearn.metrics import r2_score

def compute_r2_scores(labels_test, pred_matched, max_sources, target_dim, label_mode):
    """Compute R² scores overall, per dimension, for widths, and for positions."""
    if labels_test.shape[-1] > target_dim:
        labels_test = labels_test[..., :target_dim]
    truth_flat = labels_test.reshape(labels_test.shape[0], -1)
    pred_flat = pred_matched.reshape(pred_matched.shape[0], -1)
    
    mask = (truth_flat != 0.0) & (pred_flat != 0.0)
    
    r2_valid = r2_score(truth_flat[mask], pred_flat[mask]) if np.any(mask) else np.nan
    r2_all = r2_score(truth_flat.flatten(), pred_flat.flatten())
    
    per_dim_results = []
    for i in range(min(max_sources * target_dim, truth_flat.shape[1])):
        mask_i = (truth_flat[:, i] != 0.0) & (pred_flat[:, i] != 0.0)
        if np.sum(mask_i) > 0:
            per_dim_results.append(r2_score(truth_flat[mask_i, i], pred_flat[mask_i, i]))
        else:
            per_dim_results.append(np.nan)
            
    if label_mode == "position":
        width_indices = []
        position_indices = list(range(min(max_sources * target_dim, truth_flat.shape[1])))
    else:
        width_indices = [i for i in range(min(max_sources * target_dim, truth_flat.shape[1])) if i % 6 in [0, 2, 4]]
        position_indices = [i for i in range(min(max_sources * target_dim, truth_flat.shape[1])) if i % 6 in [1, 3, 5]]
        
    widths_t, widths_p = [], []
    for idx in width_indices:
        mask_i = (truth_flat[:, idx] != 0.0) & (pred_flat[:, idx] != 0.0)
        if np.any(mask_i):
            widths_t.extend(truth_flat[mask_i, idx])
            widths_p.extend(pred_flat[mask_i, idx])
    r2_widths = r2_score(widths_t, widths_p) if len(widths_t) > 0 else np.nan
    
    pos_t, pos_p = [], []
    for idx in position_indices:
        mask_i = (truth_flat[:, idx] != 0.0) & (pred_flat[:, idx] != 0.0)
        if np.any(mask_i):
            pos_t.extend(truth_flat[mask_i, idx])
            pos_p.extend(pred_flat[mask_i, idx])
    r2_positions = r2_score(pos_t, pos_p) if len(pos_t) > 0 else np.nan

    return {
        "r2_valid": r2_valid,
        "r2_all": r2_all,
        "per_dim_r2": per_dim_results,
        "r2_widths": r2_widths,
        "r2_positions": r2_positions
    }

def compute_mae_metrics(labels_test, pred_matched, max_sources, target_dim):
    """Compute absolute error metrics overall and per dimension."""
    truth_flat = labels_test.reshape(labels_test.shape[0], -1)
    pred_flat = pred_matched.reshape(pred_matched.shape[0], -1)
    
    mask = (truth_flat != 0.0) & (pred_flat != 0.0)
    abs_errors = np.abs(truth_flat[mask] - pred_flat[mask]) if np.any(mask) else np.array([])
    
    mae = np.mean(abs_errors) if len(abs_errors) > 0 else np.nan
    median_ae = np.median(abs_errors) if len(abs_errors) > 0 else np.nan
    min_ae = np.min(abs_errors) if len(abs_errors) > 0 else np.nan
    max_ae = np.max(abs_errors) if len(abs_errors) > 0 else np.nan
    
    per_dim_mae = []
    per_dim_median = []
    per_dim_min = []
    per_dim_max = []
    
    for i in range(min(max_sources * target_dim, truth_flat.shape[1])):
        mask_i = (truth_flat[:, i] != 0.0) & (pred_flat[:, i] != 0.0)
        if np.sum(mask_i) > 0:
            err = np.abs(truth_flat[mask_i, i] - pred_flat[mask_i, i])
            per_dim_mae.append(np.mean(err))
            per_dim_median.append(np.median(err))
            per_dim_min.append(np.min(err))
            per_dim_max.append(np.max(err))
        else:
            per_dim_mae.append(np.nan)
            per_dim_median.append(np.nan)
            per_dim_min.append(np.nan)
            per_dim_max.append(np.nan)
            
    return {
        "mae": mae,
        "median_ae": median_ae,
        "min_ae": min_ae,
        "max_ae": max_ae,
        "per_dim_mae": per_dim_mae,
        "per_dim_median_ae": per_dim_median,
        "per_dim_min_ae": per_dim_min,
        "per_dim_max_ae": per_dim_max
    }
