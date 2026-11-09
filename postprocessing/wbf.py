import numpy as np
from typing import Optional, Literal

def apply_distance_wbf(
    predictions: np.ndarray,
    dist_threshold: float = 8.0,
    conf_threshold: float = 0.35,
    k_threshold: float = 0.0,
    top_k: Optional[int] = 3,
    pad_value: float = 0.0,
) -> np.ndarray:
    """
    Applies 3D Centroid-Distance Weighted Boxes Fusion (WBF) with optional Top-K
    and Relative Confidence Thresholding.
    
    Args:
        predictions: np.ndarray of shape (batch_size, num_predictions, box_dim)
                     where box_dim is >= 7: [xw, xc, yw, yc, zw, zc, conf, ...]
        dist_threshold: Float. 3D Euclidean distance between predicted centers (xc, yc, zc)
                        below which boxes are clustered and fused into one source.
        conf_threshold: Float. Absolute minimum confidence score to consider a box.
        k_threshold: Float. Relative threshold ratio (0.0 to disable). If > 0, candidate boxes
                     must satisfy conf >= max(conf_threshold, k_threshold * max_conf_in_sample).
        top_k: Optional[int]. Maximum number of top-confidence sources to keep per sample.
        pad_value: Float. Value used to pad unoccupied slots.
        
    Returns:
        np.ndarray of shape (batch_size, num_predictions, box_dim) with fused & filtered boxes.
    """
    if predictions.shape[-1] < 7:
        raise ValueError("Predictions must have at least 7 dimensions (6 box coords + 1 conf).")
        
    batch_size, num_preds, box_dim = predictions.shape
    filtered = np.full_like(predictions, pad_value)
    
    # Centre coordinate indices in 6D box layout: [xw, xc, yw, yc, zw, zc]
    centre_idx = [1, 3, 5]
    
    for b in range(batch_size):
        batch_preds = predictions[b]
        
        # 1. Filter candidate boxes by absolute confidence and non-zero coordinates
        valid_mask = (batch_preds[:, 6] >= conf_threshold) & ~np.all(np.isclose(batch_preds[:, :6], 0.0, atol=1e-3), axis=-1)
        valid = batch_preds[valid_mask]
        if len(valid) == 0:
            continue
            
        # 2. Relative Top-K thresholding (k_threshold)
        if k_threshold > 0.0:
            s_max = np.max(valid[:, 6])
            rel_cutoff = max(conf_threshold, k_threshold * s_max)
            valid = valid[valid[:, 6] >= rel_cutoff]
            if len(valid) == 0:
                continue
                
        # 3. Sort candidates descending by confidence
        order = np.argsort(valid[:, 6])[::-1]
        valid = valid[order]
        
        # 4. Greedy 3D Distance Clustering
        clusters = []  # List of lists of box arrays
        for box in valid:
            c_box = box[centre_idx]
            matched = False
            for cluster in clusters:
                # Compute current weighted center of cluster
                confs = np.array([bx[6] for bx in cluster])
                c_cluster = np.sum(np.array([bx[centre_idx] for bx in cluster]) * confs[:, None], axis=0) / np.sum(confs)
                if np.linalg.norm(c_box - c_cluster) <= dist_threshold:
                    cluster.append(box)
                    matched = True
                    break
            if not matched:
                clusters.append([box])
                
        # 5. Fuse each cluster using confidence weights
        fused_boxes = []
        for cluster in clusters:
            if len(cluster) == 1:
                fused_boxes.append(cluster[0])
            else:
                confs = np.array([bx[6] for bx in cluster])
                tot_c = np.sum(confs)
                # Weighted average for coordinates (and any extra channels before confidence)
                coords_fused = np.sum(np.array([bx[:6] for bx in cluster]) * confs[:, None], axis=0) / tot_c
                # Fused confidence is maximum confidence in the cluster
                conf_fused = np.max(confs)
                
                fused_box = np.zeros(box_dim, dtype=predictions.dtype)
                fused_box[:6] = coords_fused
                fused_box[6] = conf_fused
                if box_dim > 7:
                    # If extra dims exist, weighted average
                    fused_box[7:] = np.sum(np.array([bx[7:] for bx in cluster]) * confs[:, None], axis=0) / tot_c
                fused_boxes.append(fused_box)
                
        # 6. Sort fused boxes by confidence descending
        fused_boxes.sort(key=lambda x: x[6], reverse=True)
        
        # 7. Apply Top-K cap
        if top_k is not None and top_k > 0:
            fused_boxes = fused_boxes[:top_k]
            
        # 8. Re-pad to original slots
        for i, box in enumerate(fused_boxes[:num_preds]):
            filtered[b, i] = box
            
    return filtered


def apply_postprocessing(
    predictions: np.ndarray,
    method: Literal["wbf", "nms", "none"] = "wbf",
    dist_threshold: float = 8.0,
    conf_threshold: float = 0.35,
    k_threshold: float = 0.0,
    top_k: Optional[int] = 3,
    iou_threshold: float = 0.1,
    pad_value: float = 0.0,
) -> np.ndarray:
    """
    Unified entry point for prediction postprocessing.
    """
    if method == "none":
        return predictions
        
    if method == "wbf":
        return apply_distance_wbf(
            predictions,
            dist_threshold=dist_threshold,
            conf_threshold=conf_threshold,
            k_threshold=k_threshold,
            top_k=top_k,
            pad_value=pad_value,
        )
        
    if method == "nms":
        from postprocessing.nms import apply_nms
        nms_preds = apply_nms(
            predictions,
            iou_threshold=iou_threshold,
            conf_threshold=conf_threshold,
            pad_value=pad_value,
        )
        if top_k is not None and top_k > 0:
            # Also apply top-k cap to NMS output if requested
            filtered = np.full_like(nms_preds, pad_value)
            for b in range(predictions.shape[0]):
                valid_mask = nms_preds[b, :, 6] >= conf_threshold
                valid = nms_preds[b, valid_mask]
                if len(valid) > 0:
                    order = np.argsort(valid[:, 6])[::-1][:top_k]
                    for i, idx in enumerate(order):
                        filtered[b, i] = valid[idx]
            return filtered
        return nms_preds
        
    raise ValueError(f"Unknown postprocessing method: {method}. Choose from 'wbf', 'nms', 'none'.")
