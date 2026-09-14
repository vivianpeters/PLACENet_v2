import tensorflow as tf
from typing import Optional
import numpy as np
from tensorflow.keras.utils import register_keras_serializable
from scipy.optimize import linear_sum_assignment
from preprocessing.load_data import label_target_dim

_EPS = 1e-7

@register_keras_serializable(package="PLACENet")
class VoxelFocalLoss(tf.keras.losses.Loss):
    """Focal BCE for sparse 3D occupancy."""

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        voxel_mask: Optional[np.ndarray] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.voxel_mask = None if voxel_mask is None else tf.constant(voxel_mask, dtype=tf.float32)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"gamma": self.gamma, "alpha": self.alpha})
        return cfg

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-7, 1.0 - 1e-7)
        ce = -y_true * tf.math.log(y_pred) - (1.0 - y_true) * tf.math.log(1.0 - y_pred)
        weight = (
            y_true * self.alpha * tf.pow(1.0 - y_pred, self.gamma)
            + (1.0 - y_true) * (1.0 - self.alpha) * tf.pow(y_pred, self.gamma)
        )
        loss = weight * ce
        if self.voxel_mask is not None:
            mask = tf.reshape(self.voxel_mask, (1, *self.voxel_mask.shape, 1))
            loss = loss * mask
            denom = tf.reduce_sum(mask) + 1e-7
            return tf.reduce_sum(loss) / denom
        return tf.reduce_mean(loss)


