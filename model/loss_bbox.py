import tensorflow as tf
import numpy as np
from typing import Tuple, Optional, Any
from tensorflow.keras.utils import register_keras_serializable
from scipy.optimize import linear_sum_assignment
from preprocessing.load_data import label_target_dim

_EPS = 1e-7

def _ensure_positive_widths(boxes):
    """Ensure positive widths for predicted boxes.
    Input format: [xwidth, xcentre, ywidth, ycentre, zwidth, zcentre]
    Output format: [xwidth, xcentre, ywidth, ycentre, zwidth, zcentre]
    """
    boxes = tf.cast(boxes, tf.float32) # Cast the boxes to float32
    xw = tf.nn.softplus(boxes[..., 0]) + 1e-6
    xcentre = boxes[..., 1] # x centre
    yw = tf.nn.softplus(boxes[..., 2]) + 1e-6
    ycentre = boxes[..., 3] # y centre
    zw = tf.nn.softplus(boxes[..., 4]) + 1e-6
    zcentre = boxes[..., 5] # z centre
    return tf.stack([xw, xcentre, yw, ycentre, zw, zcentre], axis=-1)


def _abs_gt_widths(boxes):
    """Use absolute widths for ground truth boxes.
    Input format: [xwidth, xcentre, ywidth, ycentre, zwidth, zcentre]
    Output format: [xwidth, xcentre, ywidth, ycentre, zwidth, zcentre]
    """
    boxes = tf.cast(boxes, tf.float32)
    xw = tf.abs(boxes[..., 0]) + 1e-7
    xcentre = boxes[..., 1]
    yw = tf.abs(boxes[..., 2]) + 1e-7
    ycentre = boxes[..., 3]
    zw = tf.abs(boxes[..., 4]) + 1e-7
    zcentre = boxes[..., 5]
    return tf.stack([xw, xcentre, yw, ycentre, zw, zcentre], axis=-1)


def _pairwise_smooth_l1_batch(y_true, y_pred, delta=1.0, axis_weights=None):
    """Compute pairwise smooth L1 costs for batch."""
    # y_true, y_pred: (B, S, D)
    yt = tf.expand_dims(y_true, 2)  # (B, S, 1, D)
    yp = tf.expand_dims(y_pred, 1)  # (B, 1, S, D)
    diff = yt - yp  # (B, S, S, D)
    absdiff = tf.abs(diff)
    mask_quad = tf.less_equal(absdiff, delta)
    quad = 0.5 * tf.square(absdiff)
    linear = delta * (absdiff - 0.5 * delta)
    per_elem = tf.where(mask_quad, quad, linear)
    if axis_weights is not None:
        weights = tf.constant(axis_weights, dtype=per_elem.dtype)
        per_elem = per_elem * weights
    per_box = tf.reduce_sum(per_elem, axis=-1)  # (B, S, S)
    return per_box


def _pairwise_ciou3d_batch(y_true, y_pred, eps=_EPS):
    """Compute pairwise CIoU3D *costs* for batch.
    Input format: [xwidth, xcentre, ywidth, ycentre, zwidth, zcentre]
    """
    T = _abs_gt_widths(y_true)
    P = _ensure_positive_widths(y_pred)
    # broadcast to (B, S, S, 6)
    T_exp = tf.expand_dims(T, 2)
    P_exp = tf.expand_dims(P, 1)

    xw_t = T_exp[..., 0]; xcentre_t = T_exp[..., 1]
    yw_t = T_exp[..., 2]; ycentre_t = T_exp[..., 3]
    zw_t = T_exp[..., 4]; zcentre_t = T_exp[..., 5]

    xw_p = P_exp[..., 0]; xcentre_p = P_exp[..., 1]
    yw_p = P_exp[..., 2]; ycentre_p = P_exp[..., 3]
    zw_p = P_exp[..., 4]; zcentre_p = P_exp[..., 5]

    # Convert from centre-based to min/max for IoU calculation
    xmin_t = xcentre_t - xw_t / 2.0
    xmax_t = xcentre_t + xw_t / 2.0
    ymin_t = ycentre_t - yw_t / 2.0
    ymax_t = ycentre_t + yw_t / 2.0
    zmin_t = zcentre_t - zw_t / 2.0
    zmax_t = zcentre_t + zw_t / 2.0

    xmin_p = xcentre_p - xw_p / 2.0
    xmax_p = xcentre_p + xw_p / 2.0
    ymin_p = ycentre_p - yw_p / 2.0
    ymax_p = ycentre_p + yw_p / 2.0
    zmin_p = zcentre_p - zw_p / 2.0
    zmax_p = zcentre_p + zw_p / 2.0

    # Ensure min <= max for robustness
    xmin_t_actual = tf.minimum(xmin_t, xmax_t)
    xmax_t_actual = tf.maximum(xmin_t, xmax_t)
    ymin_t_actual = tf.minimum(ymin_t, ymax_t)
    ymax_t_actual = tf.maximum(ymin_t, ymax_t)
    zmin_t_actual = tf.minimum(zmin_t, zmax_t)
    zmax_t_actual = tf.maximum(zmin_t, zmax_t)
    
    xmin_p_actual = tf.minimum(xmin_p, xmax_p)
    xmax_p_actual = tf.maximum(xmin_p, xmax_p)
    ymin_p_actual = tf.minimum(ymin_p, ymax_p)
    ymax_p_actual = tf.maximum(ymin_p, ymax_p)
    zmin_p_actual = tf.minimum(zmin_p, zmax_p)
    zmax_p_actual = tf.maximum(zmin_p, zmax_p)

    ix_min = tf.maximum(xmin_t_actual, xmin_p_actual)
    iy_min = tf.maximum(ymin_t_actual, ymin_p_actual)
    iz_min = tf.maximum(zmin_t_actual, zmin_p_actual)
    ix_max = tf.minimum(xmax_t_actual, xmax_p_actual)
    iy_max = tf.minimum(ymax_t_actual, ymax_p_actual)
    iz_max = tf.minimum(zmax_t_actual, zmax_p_actual)

    iw = tf.maximum(ix_max - ix_min, 0.0)
    ih = tf.maximum(iy_max - iy_min, 0.0)
    idp = tf.maximum(iz_max - iz_min, 0.0)
    inter = iw * ih * idp

    vol_t = xw_t * yw_t * zw_t
    vol_p = xw_p * yw_p * zw_p
    union = vol_t + vol_p - inter
    union = tf.maximum(union, eps)
    iou = inter / union
    iou = tf.clip_by_value(iou, 0.0, 1.0)

    # Calculate centers (already have them, but recalculate from actual min/max for consistency)
    cx_t = 0.5 * (xmin_t_actual + xmax_t_actual)
    cy_t = 0.5 * (ymin_t_actual + ymax_t_actual)
    cz_t = 0.5 * (zmin_t_actual + zmax_t_actual)
    cx_p = 0.5 * (xmin_p_actual + xmax_p_actual)
    cy_p = 0.5 * (ymin_p_actual + ymax_p_actual)
    cz_p = 0.5 * (zmin_p_actual + zmax_p_actual)

    dx = cx_t - cx_p
    dy = cy_t - cy_p
    dz = cz_t - cz_p
    rho2 = dx*dx + dy*dy + dz*dz

    # Enclosing box
    encl_min_x = tf.minimum(xmin_t_actual, xmin_p_actual)
    encl_min_y = tf.minimum(ymin_t_actual, ymin_p_actual)
    encl_min_z = tf.minimum(zmin_t_actual, zmin_p_actual)
    encl_max_x = tf.maximum(xmax_t_actual, xmax_p_actual)
    encl_max_y = tf.maximum(ymax_t_actual, ymax_p_actual)
    encl_max_z = tf.maximum(zmax_t_actual, zmax_p_actual)
    diag2 = tf.square(encl_max_x - encl_min_x) + tf.square(encl_max_y - encl_min_y) + tf.square(encl_max_z - encl_min_z)
    diag2 = tf.maximum(diag2, eps)

    # Compute aspect ratio consistency term v
    # Add numerical stability: avoid division by very small numbers
    width_sum_sq = xw_t**2 + yw_t**2 + zw_t**2
    width_sum_sq = tf.maximum(width_sum_sq, eps)  # Ensure >= eps
    v = ((xw_t - xw_p)**2 + (yw_t - yw_p)**2 + (zw_t - zw_p)**2) / width_sum_sq
    
    # Clip v to reasonable range to prevent extreme values while preserving relative ordering
    # This prevents alpha*v from becoming unbounded
    v = tf.clip_by_value(v, 0.0, 10.0)
    
    # Compute alpha with numerical stability
    alpha_denom = 1.0 - iou + v + eps
    alpha_denom = tf.maximum(alpha_denom, eps)  # Ensure >= eps
    alpha = v / alpha_denom
    
    # Clip alpha to reasonable range to avoid extreme values
    alpha = tf.clip_by_value(alpha, 0.0, 1.0)

    ciou = iou - (rho2 / diag2) - alpha * v
    # Only clip upper bound to 1.0 (CIoU should never exceed perfect match)
    # Allow lower bound to be more negative to properly penalize very bad matches
    # With v clipped to [0, 10] and alpha to [0, 1], worst case is approximately -11
    ciou = tf.clip_by_value(ciou, -11.0, 1.0 - eps)
    cost = 1.0 - ciou
    
    # Ensure no NaN or Inf values (replace with large cost)
    cost = tf.where(tf.math.is_finite(cost), cost, tf.constant(1e10, dtype=cost.dtype))
    
    return cost


class PLACENetCustomLoss:
    """Custom loss functions and metrics for PLACENet."""
    
    def __init__(self):
        pass

    def masked_iou_metric(self, y_true, y_pred):
        """Compute masked IoU metric."""
        return 1.0 - self.masked_iou_loss(y_true, y_pred)

    def masked_iou_loss(self, y_true, y_pred):
        """Compute masked IoU *loss* (1 - IoU).
        Input format: [xwidth, xcentre, ywidth, ycentre, zwidth, zcentre]
        """
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        # Extract only box dimensions (first 6) if using YOLO-style (7 dims)
        y_true_boxes = y_true[..., :6]
        y_pred_boxes = y_pred[..., :6]
        if y_true.shape[-1] >= 7:
            valid = y_true[..., 6] > 0.5
        else:
            valid = tf.reduce_any(tf.abs(y_true_boxes) > 1e-4, axis=-1)
        valid = tf.cast(valid, dtype=tf.float32)
        mask = tf.expand_dims(valid, -1)
        y_true_masked = y_true_boxes * mask
        y_pred_masked = y_pred_boxes * mask
        x_width_true, x_centre_true, y_width_true, y_centre_true, z_width_true, z_centre_true = tf.unstack(y_true_masked, axis=-1)
        x_width_pred, x_centre_pred, y_width_pred, y_centre_pred, z_width_pred, z_centre_pred = tf.unstack(y_pred_masked, axis=-1)
        # Convert from centre-based to min/max for IoU calculation
        x_min_true = x_centre_true - x_width_true / 2.0
        x_max_true = x_centre_true + x_width_true / 2.0
        y_min_true = y_centre_true - y_width_true / 2.0
        y_max_true = y_centre_true + y_width_true / 2.0
        z_min_true = z_centre_true - z_width_true / 2.0
        z_max_true = z_centre_true + z_width_true / 2.0
        x_min_pred = x_centre_pred - x_width_pred / 2.0
        x_max_pred = x_centre_pred + x_width_pred / 2.0
        y_min_pred = y_centre_pred - y_width_pred / 2.0
        y_max_pred = y_centre_pred + y_width_pred / 2.0
        z_min_pred = z_centre_pred - z_width_pred / 2.0
        z_max_pred = z_centre_pred + z_width_pred / 2.0
        x_intersect_min = tf.maximum(x_min_true, x_min_pred)
        x_intersect_max = tf.minimum(x_max_true, x_max_pred)
        y_intersect_min = tf.maximum(y_min_true, y_min_pred)
        y_intersect_max = tf.minimum(y_max_true, y_max_pred)
        z_intersect_min = tf.maximum(z_min_true, z_min_pred)
        z_intersect_max = tf.minimum(z_max_true, z_max_pred)
        intersection = tf.maximum(x_intersect_max - x_intersect_min, 0) * tf.maximum(y_intersect_max - y_intersect_min, 0) * tf.maximum(z_intersect_max - z_intersect_min, 0)
        true_volume = tf.abs(x_width_true) * tf.abs(y_width_true) * tf.abs(z_width_true)
        pred_volume = tf.abs(x_width_pred) * tf.abs(y_width_pred) * tf.abs(z_width_pred)
        union = true_volume + pred_volume - intersection + 1e-7
        iou = intersection / union
        
        # Only average over valid (non-padded) boxes
        valid_box_mask = valid
        
        # Compute mean IoU only over valid boxes
        iou_masked = iou * valid_box_mask
        sum_iou = tf.reduce_sum(iou_masked)
        count_valid = tf.reduce_sum(valid_box_mask)
        mean_iou = tf.cond(
            count_valid > 0,
            lambda: sum_iou / count_valid,
            lambda: 0.0
        )
        
        return 1.0 - mean_iou


@tf.keras.utils.register_keras_serializable(package="PLACENet")
class BBoxMatchingLoss(tf.keras.losses.Loss):
    """
    YOLO-style loss that combines box regression with confidence prediction.
    Output format: (max_sources, 7) where last dimension is [boxes(6), confidence(1)]
    Uses matching (greedy or JV) for box assignment.
    """
    def __init__(
        self,
        base: str = "hybrid", # "smooth_l1", "ciou", or "hybrid"
        max_sources: int = 5, # Maximum number of sources to classify
        matching_strategy: str = "jv", # "greedy" or "jv"
        delta: float = 1.0, # Delta for the smooth L1 loss
        ciou_weight: float = 1.0, # Weight for the CIoU loss
        smooth_weight: float = 1.0, # Weight for the smooth L1 loss
        confidence_weight: float = 1.5, # Weight for the confidence loss
        noobj_weight: float = 0.1, # Weight for the no object loss
        label_mode: str = "bbox",
        position_axis_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        name: str = "yolo_style_matching_loss", # Name of the loss since Keras needs a name
        **kwargs,
    ):
        self.init_log_var_box = float(kwargs.pop("init_log_var_box", 0.0))
        self.init_log_var_conf = float(kwargs.pop("init_log_var_conf", 0.0))
        super().__init__(name=name, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE, **kwargs)
        assert label_mode in ("bbox", "position")
        if label_mode == "position" and base != "smooth_l1":
            print(
                f"[BBoxMatchingLoss] label_mode=position: forcing base='smooth_l1' "
                f"(was {base!r}; CIoU requires box widths)"
            )
            base = "smooth_l1"
        assert base in ("smooth_l1", "ciou", "hybrid")
        assert matching_strategy in ("greedy", "jv")
        self.base = base
        self.max_sources = int(max_sources)
        self.matching_strategy = matching_strategy
        self.delta = float(delta)
        self.ciou_weight = float(ciou_weight)
        self.smooth_weight = float(smooth_weight)
        self.confidence_weight = float(confidence_weight)
        self.noobj_weight = float(noobj_weight)
        self.label_mode = label_mode
        self.target_dim = label_target_dim(label_mode)
        self.position_axis_weights = tuple(float(w) for w in position_axis_weights)
        ## Update: use homoscedastic uncertainty weighting (Kendall et al., CVPR 2018)
        self.s_box = tf.Variable(self.init_log_var_box, trainable=True, dtype=tf.float32, name="s_box")
        self.s_conf = tf.Variable(self.init_log_var_conf, trainable=True, dtype=tf.float32, name="s_conf")
        self.s_box.regularizer = None
        self.s_conf.regularizer = None
        matching_type = "JV (Jonker-Volgenant)"
        axis_msg = ""
        if label_mode == "position" and self.position_axis_weights != (1.0, 1.0, 1.0):
            axis_msg = f", position_axis_weights={self.position_axis_weights}"
        print(
            f"[BBoxMatchingLoss] Initialized with matching: {matching_type}, "
            f"label_mode={label_mode}, base={base}, confidence_weight={confidence_weight}, "
            f"noobj_weight={noobj_weight}{axis_msg}"
        )

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "base": self.base,
            "max_sources": self.max_sources,
            "delta": self.delta,
            "ciou_weight": self.ciou_weight,
            "smooth_weight": self.smooth_weight,
            "confidence_weight": self.confidence_weight,
            "noobj_weight": self.noobj_weight,
            "label_mode": self.label_mode,
            "position_axis_weights": self.position_axis_weights,
            "init_log_var_box": float(self.s_box.numpy()),
            "init_log_var_conf": float(self.s_conf.numpy()),
        })
        return cfg

    def _split_outputs(self, y_pred):
        """Split predictions into targets and confidence."""
        pred_boxes_batch = y_pred[..., :self.target_dim]
        pred_conf_batch = y_pred[..., self.target_dim]
        return pred_boxes_batch, pred_conf_batch

    def _confidence_loss(self, conf_true, conf_pred):
        """Compute YOLO-style confidence loss using Focal Loss."""
        gamma = 0.0  # Setting gamma=0.0 mathematically turns this into standard BCE!
        conf_pred = tf.clip_by_value(conf_pred, 1e-7, 1.0 - 1e-7)
        
        # Object focal loss (when conf_true=1.0)
        obj_loss = -conf_true * tf.pow(1.0 - conf_pred, gamma) * tf.math.log(conf_pred)
        
        # No-object focal loss (when conf_true=0.0)
        noobj_loss = -(1.0 - conf_true) * tf.pow(conf_pred, gamma) * tf.math.log(1.0 - conf_pred)
        
        # Combine with different weights
        confidence_loss = obj_loss + self.noobj_weight * noobj_loss
        
        return tf.reduce_mean(confidence_loss)

    def _mask_costs(self, cost, true_mask, pred_mask):
        """Mask invalid entries in cost matrix."""
        inf = tf.constant(1e10, dtype=cost.dtype)
        tm = tf.logical_not(true_mask)
        pm = tf.logical_not(pred_mask)
        row_mask = tf.cast(tm, cost.dtype) * inf
        col_mask = tf.cast(pm, cost.dtype) * inf
        cost = cost + tf.expand_dims(row_mask, 2) + tf.expand_dims(col_mask, 1)
        return cost

    def _jv_numpy_batch_indices(self, cost_batch_np):
        """
        Batch-wise JV (Jonker-Volgenant) matching that returns INDICES.
        This matches the evaluation implementation using linear_sum_assignment.
        
        Args:
            cost_batch_np: (B, S, S) numpy array of costs
        Returns:
            indices: (B, S, 3) array where indices[b, k, :] = [b, row, col] for k-th match in batch b
            counts: (B,) array of number of valid matches per sample
        """
        B, S, _ = cost_batch_np.shape
        indices = np.full((B, S, 3), -1, dtype=np.int32)  # [batch_idx, row, col]
        counts = np.zeros((B,), dtype=np.int32)
        
        for b in range(B):
            cm = cost_batch_np[b]  # (S, S)
            
            # If all entries are large (invalid), skip this sample
            if np.isfinite(cm).sum() == 0 or cm.size == 0:
                continue
            
            try:
                # Use linear_sum_assignment (Jonker-Volgenant algorithm)
                row_ind, col_ind = linear_sum_assignment(cm)
            except Exception as e:
                print(f"Error in linear_sum_assignment for batch {b}: {e}")
                # Fallback to greedy in numpy
                row_ind = []
                col_ind = []
                cm_copy = cm.copy()
                K = min(cm.shape)
                for _ in range(K):
                    idx = np.unravel_index(np.argmin(cm_copy, axis=None), cm_copy.shape)
                    row_ind.append(idx[0])
                    col_ind.append(idx[1])
                    cm_copy[idx[0], :] = 1e10
                    cm_copy[:, idx[1]] = 1e10
                row_ind = np.array(row_ind, dtype=np.int32)
                col_ind = np.array(col_ind, dtype=np.int32)
            
            # Filter out masked assignments (where cost is very large)
            if len(row_ind) > 0:
                valid_mask = np.isfinite(cm[row_ind, col_ind]) & (cm[row_ind, col_ind] < 1e9)
                row_ind = row_ind[valid_mask]
                col_ind = col_ind[valid_mask]
                
                n_matches = len(row_ind)
                if n_matches > 0:
                    indices[b, :n_matches, 0] = b
                    indices[b, :n_matches, 1] = row_ind
                    indices[b, :n_matches, 2] = col_ind
                    counts[b] = n_matches
        
        return indices, counts

    def _jv_tf_batch(self, cost_batch, match_cost=None):
        """
        Batch-wise JV matching that maintains gradient flow.
        Uses numpy for matching but gathers from TensorFlow tensor for gradients.
        """
        B = tf.shape(cost_batch)[0]
        S = self.max_sources
        
        if match_cost is None:
            match_cost = cost_batch
            
        # Get indices from numpy JV (non-differentiable)
        indices_np, counts_np = tf.numpy_function(
            self._jv_numpy_batch_indices,
            [match_cost],
            [tf.int32, tf.int32]
        )
        indices_np.set_shape((None, S, 3))
        counts_np.set_shape((None,))
        
        def per_sample_gather(args):
            """Gather matched costs for one sample"""
            sample_idx, sample_indices, count = args
            
            valid_indices = sample_indices[:count]
            
            def no_matches():
                return tf.constant(0.0, dtype=tf.float32)
            
            def has_matches():
                rows = valid_indices[:, 1]
                cols = valid_indices[:, 2]
                gather_indices = tf.stack([rows, cols], axis=1)
                costs = tf.gather_nd(cost_batch[sample_idx], gather_indices)
                costs = tf.where(tf.math.is_finite(costs), costs, tf.constant(0.0, dtype=costs.dtype))
                return tf.reduce_mean(costs)
            
            return tf.cond(tf.equal(count, 0), no_matches, has_matches)
        
        sample_indices = tf.range(B, dtype=tf.int32)
        mean_costs = tf.map_fn(
            per_sample_gather,
            (sample_indices, indices_np, counts_np),
            fn_output_signature=tf.TensorSpec(shape=(), dtype=tf.float32)
        )
        return mean_costs, indices_np, counts_np

    def _match_conf_tf(self, conf_true, indices, counts):
        """
        Construct matched confidence targets (B,S) based on dynamic matching indices.
        Prediction slot k matched to GT row j inherits conf_true[b,j].
        Unmatched prediction slots remain 0.
        """
        B = tf.shape(conf_true)[0]
        S = self.max_sources

        def per_sample_target(args):
            sample_idx, sample_matches, count = args
            valid_indices = sample_matches[:count]
            
            def no_matches():
                return tf.zeros((S,), dtype=tf.float32)
            
            def has_matches():
                gt_rows = valid_indices[:, 1]
                pred_cols = valid_indices[:, 2]
                gt_conf_vals = tf.gather(conf_true[sample_idx], gt_rows)
                scatter_idx = tf.expand_dims(pred_cols, axis=-1)
                return tf.scatter_nd(scatter_idx, gt_conf_vals, shape=(S,))
            
            return tf.cond(tf.equal(count, 0), no_matches, has_matches)
        
        batch_indices = tf.range(B, dtype=tf.int32)
        matched_conf = tf.map_fn(
            per_sample_target,
            (batch_indices, indices, counts),
            fn_output_signature=tf.TensorSpec(shape=(S,), dtype=tf.float32)
        )
        return matched_conf

    def call(self, y_true, y_pred):
        """
        Compute YOLO-style loss with matching.
        
        Args:
            y_true: (B, S, target_dim+1) where last dim is [targets, confidence(1)]
            y_pred: (B, S, target_dim+1) where last dim is [targets, confidence(1)]
        """
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        B = tf.shape(y_true)[0]
        S = self.max_sources

        # Split into targets and confidence
        targets_true, conf_true = self._split_outputs(y_true)
        targets_pred, conf_pred = self._split_outputs(y_pred)

        # Compute valid masks from confidence
        true_mask = conf_true > 0.5
        # Predictions are always considered valid for matching, matching handles extra slots
        pred_mask = tf.ones_like(conf_pred, dtype=tf.bool)

        targets_t_clip = targets_true[:, :S, :]
        targets_p_clip = targets_pred[:, :S, :]

        # Compute pairwise costs for box matching
        axis_weights = self.position_axis_weights if self.label_mode == "position" else None
        if self.base in ("smooth_l1", "hybrid"):
            cost_smooth = _pairwise_smooth_l1_batch(
                targets_t_clip, targets_p_clip, delta=self.delta, axis_weights=axis_weights
            )
        else:
            cost_smooth = tf.zeros((B, S, S), dtype=tf.float32)

        if self.base in ("ciou", "hybrid") and self.label_mode == "bbox":
            cost_ciou = _pairwise_ciou3d_batch(targets_t_clip, targets_p_clip)
        else:
            cost_ciou = tf.zeros((B, S, S), dtype=tf.float32)

        cost_pure = self.smooth_weight * cost_smooth + self.ciou_weight * cost_ciou # Combine the smooth L1 and CIoU costs
        
        # Include predicted confidence in the matching cost (Higher confidence = Lower cost)
        # Dynamically scale confidence penalty based on average spatial cost magnitude so confidence
        # and spatial distance are proportionally balanced without hardcoded magic numbers.
        cost_scale = tf.stop_gradient(tf.reduce_mean(cost_pure) + 1.0)
        conf_penalty = tf.expand_dims(conf_pred, axis=1) # (B, 1, S)
        true_mask_float = tf.cast(tf.expand_dims(true_mask, axis=2), tf.float32) # (B, S, 1)
        match_cost = cost_pure - (cost_scale * self.confidence_weight * conf_penalty * true_mask_float)
        match_cost = self._mask_costs(match_cost, true_mask, pred_mask)

        # Check for valid pairs
        any_true = tf.reduce_any(true_mask, axis=1)
        any_pair = tf.logical_and(any_true, tf.reduce_any(pred_mask, axis=1))

        # Box matching loss (using JV)
        box_loss_per_sample, match_indices, match_counts = self._jv_tf_batch(cost_pure, match_cost)

        box_loss_per_sample = tf.where(any_pair, box_loss_per_sample, tf.zeros_like(box_loss_per_sample))

        # Confidence loss
        matched_conf = self._match_conf_tf(conf_true, match_indices, match_counts)
        confidence_loss = self._confidence_loss(matched_conf, conf_pred)

        # Combine losses using homoscedastic uncertainty
        box_loss_mean = tf.reduce_mean(box_loss_per_sample)
        s_box = self.s_box
        s_conf = self.s_conf
        
        # Using Kendall et al., CVPR 2018 formulation
        # Regression: 0.5 * exp(-s) * L + 0.5 * s
        # Classification: exp(-s) * L + 0.5 * s
        #total_loss = (
        #    0.5 * tf.exp(-s_box) * box_loss_mean + 0.5 * s_box +
        #    tf.exp(-s_conf) * confidence_loss + 0.5 * s_conf
        #)

        total_loss = box_loss_mean + self.confidence_weight * confidence_loss
        
        total_loss = tf.where(tf.math.is_finite(total_loss), total_loss, tf.constant(0.0, dtype=total_loss.dtype))
        return total_loss


