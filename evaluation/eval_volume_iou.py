import numpy as np

def compute_total_volume_iou(y_true, y_pred, binx=102, biny=102, binz=72, pad_value=-1):
    """
    Computes the total volume IoU for a batch of predictions vs truth.
    Converts all predicted boxes and all true boxes into two 3D boolean grids
    and computes the Intersection over Union of the grids.
    
    Args:
        y_true: np.ndarray of shape (batch_size, max_sources, box_dim)
        y_pred: np.ndarray of shape (batch_size, max_sources, box_dim)
        binx, biny, binz: Dimensions of the voxel grid
        pad_value: Value used to pad empty slots
        
    Returns:
        total_volume_ious: List of IoU scores for each item in the batch
    """
    batch_size = y_true.shape[0]
    total_volume_ious = []
    
    for b in range(batch_size):
        actual_hist = np.zeros((binx, biny, binz), dtype=np.int8)
        prediction_hist = np.zeros((binx, biny, binz), dtype=np.int8)
        
        # Process predictions
        for source_idx in range(y_pred.shape[1]):
            source_bbox = y_pred[b, source_idx]
            if np.all(source_bbox == pad_value):
                continue
            
            xwidth, xcentre, ywidth, ycentre, zwidth, zcentre = source_bbox[:6]
            xmin = max(0, int(xcentre - xwidth / 2.0))
            xmax = min(binx - 1, int(xcentre + xwidth / 2.0))
            ymin = max(0, int(ycentre - ywidth / 2.0))
            ymax = min(biny - 1, int(ycentre + ywidth / 2.0))
            zmin = max(0, int(zcentre - zwidth / 2.0))
            zmax = min(binz - 1, int(zcentre + zwidth / 2.0))
            
            if xmin <= xmax and ymin <= ymax and zmin <= zmax:
                prediction_hist[xmin:xmax+1, ymin:ymax+1, zmin:zmax+1] = 1
                
        # Process actuals
        for source_idx in range(y_true.shape[1]):
            source_bbox = y_true[b, source_idx]
            if np.all(source_bbox == pad_value):
                continue
            
            xwidth, xcentre, ywidth, ycentre, zwidth, zcentre = source_bbox[:6]
            xmin = max(0, int(xcentre - xwidth / 2.0))
            xmax = min(binx - 1, int(xcentre + xwidth / 2.0))
            ymin = max(0, int(ycentre - ywidth / 2.0))
            ymax = min(biny - 1, int(ycentre + ywidth / 2.0))
            zmin = max(0, int(zcentre - zwidth / 2.0))
            zmax = min(binz - 1, int(zcentre + zwidth / 2.0))
            
            if xmin <= xmax and ymin <= ymax and zmin <= zmax:
                actual_hist[xmin:xmax+1, ymin:ymax+1, zmin:zmax+1] = 1
                
        intersection = np.sum((actual_hist > 0) & (prediction_hist > 0))
        union = np.sum((actual_hist > 0) | (prediction_hist > 0))
        total_volume_iou = float(intersection / union) if union > 0 else 0.0
        total_volume_ious.append(total_volume_iou)
        
    return total_volume_ious
