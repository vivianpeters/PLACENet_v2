
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import register_keras_serializable
from scipy.optimize import linear_sum_assignment
from preprocessing.load_data import label_target_dim

_EPS = 1e-7

@register_keras_serializable(package="PLACENet")
def _transform_box_params(box_tensor, center_min=None, center_scale=None):
    """
    Convert raw box head outputs into valid box parameters.

    Args:
        box_tensor: Tensor (..., 6) in [xw, xc, yw, yc, zw, zc] layout.
        center_min: Optional length-3 iterable for x/y/z minimum centres.
        center_scale: Optional length-3 iterable for x/y/z centre ranges.
    """
    widths = tf.nn.softplus(tf.gather(box_tensor, [0, 2, 4], axis=-1))
    centers_unit = tf.math.sigmoid(tf.gather(box_tensor, [1, 3, 5], axis=-1))

    if center_min is None:
        center_min = [0.0, 0.0, 0.0]
    if center_scale is None:
        center_scale = [1.0, 1.0, 1.0]

    center_min_const = tf.constant(center_min, dtype=box_tensor.dtype)
    center_scale_const = tf.constant(center_scale, dtype=box_tensor.dtype)
    centers = centers_unit * center_scale_const + center_min_const

    return tf.stack(
        [
            widths[..., 0], centers[..., 0],
            widths[..., 1], centers[..., 1],
            widths[..., 2], centers[..., 2],
        ],
        axis=-1,
    )

@register_keras_serializable(package="PLACENet")
def _transform_position_params(position_tensor, center_min=None, center_scale=None):
    """Map raw position head outputs into physical (x, y, z) coordinates."""
    centers_unit = tf.math.sigmoid(position_tensor)

    if center_min is None:
        center_min = [0.0, 0.0, 0.0]
    if center_scale is None:
        center_scale = [1.0, 1.0, 1.0]

    center_min_const = tf.constant(center_min, dtype=position_tensor.dtype)
    center_scale_const = tf.constant(center_scale, dtype=position_tensor.dtype)
    return centers_unit * center_scale_const + center_min_const

