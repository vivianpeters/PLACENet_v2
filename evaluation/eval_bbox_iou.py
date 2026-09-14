import numpy as np
from preprocessing.load_data import infer_label_mode, label_target_dim, label_output_dim

def compute_iou_per_slot_stats(y_true, y_pred, label_mode="bbox"):
    """
    Compute 3D IoU per valid slot from labels and predictions (NumPy only).
    Box format: [xwidth, xcentre, ywidth, ycentre, zwidth, zcentre].
    Returns (mean_iou, std_iou, count) over valid (non-padded) slots.
    """
    if label_mode == "position" or infer_label_mode(last_dim=y_true.shape[-1]) == "position":
        return np.nan, np.nan, 0

    target_dim = label_target_dim("bbox")
    output_dim = label_output_dim("bbox")
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape[-1] == output_dim:
        y_true = y_true[..., :target_dim]
    if y_pred.shape[-1] == output_dim:
        y_pred = y_pred[..., :target_dim]
    # Valid slot = all 6 dims non-pad for BOTH true and pred (True Positives only)
    valid = np.all(y_true != 0.0, axis=-1) & np.all(y_pred != 0.0, axis=-1)
    # Centre/width -> min/max
    xw_t, xc_t, yw_t, yc_t, zw_t, zc_t = y_true[..., 0], y_true[..., 1], y_true[..., 2], y_true[..., 3], y_true[..., 4], y_true[..., 5]
    xw_p, xc_p, yw_p, yc_p, zw_p, zc_p = y_pred[..., 0], y_pred[..., 1], y_pred[..., 2], y_pred[..., 3], y_pred[..., 4], y_pred[..., 5]
    x_min_t = xc_t - np.abs(xw_t) / 2.0
    x_max_t = xc_t + np.abs(xw_t) / 2.0
    y_min_t = yc_t - np.abs(yw_t) / 2.0
    y_max_t = yc_t + np.abs(yw_t) / 2.0
    z_min_t = zc_t - np.abs(zw_t) / 2.0
    z_max_t = zc_t + np.abs(zw_t) / 2.0
    x_min_p = xc_p - np.abs(xw_p) / 2.0
    x_max_p = xc_p + np.abs(xw_p) / 2.0
    y_min_p = yc_p - np.abs(yw_p) / 2.0
    y_max_p = yc_p + np.abs(yw_p) / 2.0
    z_min_p = zc_p - np.abs(zw_p) / 2.0
    z_max_p = zc_p + np.abs(zw_p) / 2.0
    xi_min = np.maximum(x_min_t, x_min_p)
    xi_max = np.minimum(x_max_t, x_max_p)
    yi_min = np.maximum(y_min_t, y_min_p)
    yi_max = np.minimum(y_max_t, y_max_p)
    zi_min = np.maximum(z_min_t, z_min_p)
    zi_max = np.minimum(z_max_t, z_max_p)
    inter = np.maximum(xi_max - xi_min, 0) * np.maximum(yi_max - yi_min, 0) * np.maximum(zi_max - zi_min, 0)
    vol_t = np.abs(xw_t) * np.abs(yw_t) * np.abs(zw_t)
    vol_p = np.abs(xw_p) * np.abs(yw_p) * np.abs(zw_p)
    union = vol_t + vol_p - inter
    iou = inter / np.maximum(union, 1e-12)
    valid_ious = iou[valid]
    if len(valid_ious) == 0:
        return np.nan, np.nan, 0
    return np.mean(valid_ious), np.std(valid_ious), len(valid_ious)

def get_iou_by_source_count(y_true, y_pred, label_mode="bbox"):
    """
    Computes IoU for each valid slot and associates it with the true source count 
    for that sample. Returns a list of dicts: [{'True Source Count': c, 'IoU': iou}, ...].
    """
    if label_mode == "position" or infer_label_mode(last_dim=y_true.shape[-1]) == "position":
        return []

    target_dim = label_target_dim("bbox")
    output_dim = label_output_dim("bbox")
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape[-1] == output_dim:
        y_true = y_true[..., :target_dim]
    if y_pred.shape[-1] == output_dim:
        y_pred = y_pred[..., :target_dim]
    
    # Valid slot = all 6 dims non-pad for BOTH true and pred (True Positives only)
    valid = np.all(y_true != 0.0, axis=-1) & np.all(y_pred != 0.0, axis=-1)
    
    # True sources count per sample
    true_mask = ~np.all(y_true == 0.0, axis=-1)
    true_counts = np.sum(true_mask, axis=-1)
    
    xw_t, xc_t, yw_t, yc_t, zw_t, zc_t = y_true[..., 0], y_true[..., 1], y_true[..., 2], y_true[..., 3], y_true[..., 4], y_true[..., 5]
    xw_p, xc_p, yw_p, yc_p, zw_p, zc_p = y_pred[..., 0], y_pred[..., 1], y_pred[..., 2], y_pred[..., 3], y_pred[..., 4], y_pred[..., 5]
    x_min_t, x_max_t = xc_t - np.abs(xw_t) / 2.0, xc_t + np.abs(xw_t) / 2.0
    y_min_t, y_max_t = yc_t - np.abs(yw_t) / 2.0, yc_t + np.abs(yw_t) / 2.0
    z_min_t, z_max_t = zc_t - np.abs(zw_t) / 2.0, zc_t + np.abs(zw_t) / 2.0
    
    x_min_p, x_max_p = xc_p - np.abs(xw_p) / 2.0, xc_p + np.abs(xw_p) / 2.0
    y_min_p, y_max_p = yc_p - np.abs(yw_p) / 2.0, yc_p + np.abs(yw_p) / 2.0
    z_min_p, z_max_p = zc_p - np.abs(zw_p) / 2.0, zc_p + np.abs(zw_p) / 2.0
    
    xi_min, xi_max = np.maximum(x_min_t, x_min_p), np.minimum(x_max_t, x_max_p)
    yi_min, yi_max = np.maximum(y_min_t, y_min_p), np.minimum(y_max_t, y_max_p)
    zi_min, zi_max = np.maximum(z_min_t, z_min_p), np.minimum(z_max_t, z_max_p)
    
    inter = np.maximum(xi_max - xi_min, 0) * np.maximum(yi_max - yi_min, 0) * np.maximum(zi_max - zi_min, 0)
    vol_t = np.abs(xw_t) * np.abs(yw_t) * np.abs(zw_t)
    vol_p = np.abs(xw_p) * np.abs(yw_p) * np.abs(zw_p)
    union = vol_t + vol_p - inter
    iou = inter / np.maximum(union, 1e-12)
    
    results = []
    for i in range(len(y_true)):
        count = int(true_counts[i])
        for j in range(len(y_true[i])):
            if valid[i, j]:
                results.append({"True Source Count": count, "IoU": iou[i, j]})
    return results