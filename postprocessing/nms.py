import numpy as np

def compute_box_3d_diou(box1, box2):
    """
    Computes the 3D Distance Intersection over Union (DIoU) of two bounding boxes.
    Assumes boxes are in format: [xwidth, xcentre, ywidth, ycentre, zwidth, zcentre]
    """
    xw1, xc1, yw1, yc1, zw1, zc1 = box1[:6]
    xw2, xc2, yw2, yc2, zw2, zc2 = box2[:6]
    
    # Calculate min and max coordinates
    x1_min, x1_max = xc1 - xw1/2, xc1 + xw1/2
    y1_min, y1_max = yc1 - yw1/2, yc1 + yw1/2
    z1_min, z1_max = zc1 - zw1/2, zc1 + zw1/2
    
    x2_min, x2_max = xc2 - xw2/2, xc2 + xw2/2
    y2_min, y2_max = yc2 - yw2/2, yc2 + yw2/2
    z2_min, z2_max = zc2 - zw2/2, zc2 + zw2/2
    
    # Intersection dimensions
    inter_x_min = max(x1_min, x2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_min = max(y1_min, y2_min)
    inter_y_max = min(y1_max, y2_max)
    inter_z_min = max(z1_min, z2_min)
    inter_z_max = min(z1_max, z2_max)
    
    # Intersection volume
    inter_x = max(0, inter_x_max - inter_x_min)
    inter_y = max(0, inter_y_max - inter_y_min)
    inter_z = max(0, inter_z_max - inter_z_min)
    intersection = inter_x * inter_y * inter_z
    
    # Union volume
    vol1 = max(0, xw1) * max(0, yw1) * max(0, zw1)
    vol2 = max(0, xw2) * max(0, yw2) * max(0, zw2)
    union = vol1 + vol2 - intersection
    
    if union <= 0:
        iou = 0.0
    else:
        iou = intersection / union
        
    # Center distance squared (d^2)
    d2 = (xc1 - xc2)**2 + (yc1 - yc2)**2 + (zc1 - zc2)**2
    
    # Smallest enclosing box
    enc_x_min = min(x1_min, x2_min)
    enc_x_max = max(x1_max, x2_max)
    enc_y_min = min(y1_min, y2_min)
    enc_y_max = max(y1_max, y2_max)
    enc_z_min = min(z1_min, z2_min)
    enc_z_max = max(z1_max, z2_max)
    
    # Diagonal length squared of enclosing box (c^2)
    c2 = (enc_x_max - enc_x_min)**2 + (enc_y_max - enc_y_min)**2 + (enc_z_max - enc_z_min)**2
    
    if c2 == 0:
        return iou
        
    diou = iou - (d2 / c2)
    return diou

def apply_nms(predictions, iou_threshold=0.5, conf_threshold=0.5, pad_value=-1):
    """
    Applies Non-Maximum Suppression to a batch of predictions using DIoU.
    
    Args:
        predictions: np.ndarray of shape (batch_size, num_predictions, box_dim)
                     where box_dim is at least 7 (6 coordinates + 1 confidence score)
        iou_threshold: Float. Boxes with DIoU > threshold will be suppressed.
        conf_threshold: Float. Boxes with confidence < threshold will be discarded immediately.
        pad_value: Value used to pad the output back to num_predictions slots.
        
    Returns:
        np.ndarray of shape (batch_size, num_predictions, box_dim) with suppressed boxes padded.
    """
    if predictions.shape[-1] < 7:
        raise ValueError("Predictions must have at least 7 dimensions (6 box + 1 conf) for NMS.")
        
    batch_size = predictions.shape[0]
    num_preds = predictions.shape[1]
    box_dim = predictions.shape[-1]
    
    filtered_predictions = np.full_like(predictions, pad_value)
    
    for b in range(batch_size):
        batch_preds = predictions[b]
        
        # 1. Filter by confidence and validity
        valid_mask = (batch_preds[:, 6] >= conf_threshold) & ~np.all(np.isclose(batch_preds[..., :6], pad_value, atol=1e-3), axis=-1)
        valid_boxes = batch_preds[valid_mask]
        
        if len(valid_boxes) == 0:
            continue
            
        # 2. Sort by confidence descending
        sorted_indices = np.argsort(valid_boxes[:, 6])[::-1]
        sorted_boxes = valid_boxes[sorted_indices]
        
        kept_boxes = []
        
        # 3. Suppress overlapping boxes
        while len(sorted_boxes) > 0:
            # Pop the most confident box
            current_box = sorted_boxes[0]
            kept_boxes.append(current_box)
            
            if len(sorted_boxes) == 1:
                break
                
            # Compute DIoU between current_box and all remaining boxes
            remaining_boxes = sorted_boxes[1:]
            dious = np.array([compute_box_3d_diou(current_box, box) for box in remaining_boxes])
            
            # Keep only boxes that do NOT overlap too much with the current box
            keep_mask = dious <= iou_threshold
            sorted_boxes = remaining_boxes[keep_mask]
            
        # 4. Re-pad to the original sequence length
        for i, box in enumerate(kept_boxes):
            if i < num_preds:
                filtered_predictions[b, i] = box
                
    return filtered_predictions
