import numpy as np
from sklearn.metrics import r2_score
from preprocessing.load_data import label_target_dim, label_output_dim


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
        
    truth_reshaped = labels_test.reshape(-1, target_dim)
    pred_reshaped = pred_matched.reshape(-1, target_dim)
    
    valid_rows = ~np.all(truth_reshaped == 0.0, axis=-1) & ~np.all(pred_reshaped == 0.0, axis=-1)
    
    if not np.any(valid_rows):
        return {
            "r2_valid": np.nan,
            "r2_all": np.nan,
            "per_dim_r2": [np.nan]*target_dim,
            "r2_widths": np.nan,
            "r2_positions": np.nan
        }
        
    truth_valid = truth_reshaped[valid_rows]
    pred_valid = pred_reshaped[valid_rows]
    
    r2_valid = r2_score(truth_valid, pred_valid, multioutput='variance_weighted')
    r2_all = r2_score(truth_reshaped, pred_reshaped, multioutput='variance_weighted')
    
    per_dim_results = []
    for i in range(target_dim):
        r2_dim = r2_score(truth_valid[:, i], pred_valid[:, i])
        per_dim_results.append(r2_dim)
            
    if label_mode == "bbox":
        widths_t = truth_valid[:, [0, 2, 4]]
        widths_p = pred_valid[:, [0, 2, 4]]
        r2_widths = r2_score(widths_t, widths_p, multioutput='variance_weighted')
        
        pos_t = truth_valid[:, [1, 3, 5]]
        pos_p = pred_valid[:, [1, 3, 5]]
        r2_positions = r2_score(pos_t, pos_p, multioutput='variance_weighted')
    else:
        pos_t = truth_valid[:, [0, 1, 2]]
        pos_p = pred_valid[:, [0, 1, 2]]
        r2_positions = r2_score(pos_t, pos_p, multioutput='variance_weighted')
        r2_widths = np.nan

    return {
        "r2_valid": r2_valid,
        "r2_all": r2_all,
        "per_dim_r2": per_dim_results,
        "r2_widths": r2_widths,
        "r2_positions": r2_positions
    }

def compute_mae_metrics(labels_test, pred_matched, max_sources, target_dim, label_mode):
    """Compute absolute error metrics overall and per dimension."""
    if labels_test.shape[-1] > target_dim:
        labels_test = labels_test[..., :target_dim]
        
    truth_reshaped = labels_test.reshape(-1, target_dim)
    pred_reshaped = pred_matched.reshape(-1, target_dim)
    
    valid_rows = ~np.all(truth_reshaped == 0.0, axis=-1) & ~np.all(pred_reshaped == 0.0, axis=-1)
    
    if not np.any(valid_rows):
        return {
            "mean_3d_dist": np.nan,
            "mae_pos": np.nan,
            "mae_widths": np.nan,
            "count": 0,
            "mae": np.nan,
            "mae_std": np.nan
        }
        
    truth_valid = truth_reshaped[valid_rows]
    pred_valid = pred_reshaped[valid_rows]
    
    abs_errors = np.abs(truth_valid - pred_valid)
    mean_mae = np.mean(abs_errors)
    std_mae = np.std(abs_errors)
    
    if label_mode == "bbox":
        pos_t = truth_valid[:, [1, 3, 5]]
        pos_p = pred_valid[:, [1, 3, 5]]
        widths_t = truth_valid[:, [0, 2, 4]]
        widths_p = pred_valid[:, [0, 2, 4]]
        
        dist_3d = np.linalg.norm(pos_t - pos_p, axis=1) * 10.0 # to mm
        mean_3d_dist = np.mean(dist_3d)
        
        mae_pos = np.mean(np.abs(pos_t - pos_p))
        mae_widths = np.mean(np.abs(widths_t - widths_p))
    else:
        pos_t = truth_valid[:, [0, 1, 2]]
        pos_p = pred_valid[:, [0, 1, 2]]
        
        dist_3d = np.linalg.norm(pos_t - pos_p, axis=1) * 10.0 # to mm
        mean_3d_dist = np.mean(dist_3d)
        
        mae_pos = np.mean(np.abs(pos_t - pos_p))
        mae_widths = np.nan
        
    return {
        "mae": mean_mae,
        "mae_std": std_mae,
        "mean_3d_dist": mean_3d_dist,
        "mae_pos": mae_pos,
        "mae_widths": mae_widths,
        "count": np.sum(valid_rows)
    }
