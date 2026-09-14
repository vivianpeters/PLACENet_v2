import tensorflow as tf
import numpy as np
from typing import Tuple, List, Optional, Any
from tensorflow.keras.utils import register_keras_serializable
from scipy.optimize import linear_sum_assignment
from preprocessing.load_data import label_target_dim

_EPS = 1e-7

def _upscale_5_to_10(shared_features: tf.Tensor, l2: tf.keras.regularizers.Regularizer) -> tf.Tensor:
    """Latent 5x5x5 -> 10x10x10 via 2x upsampling + conv."""
    x = tf.keras.layers.Dense(5 * 5 * 5 * 32, activation="relu")(shared_features)
    x = tf.keras.layers.Reshape((5, 5, 5, 32))(x)
    x = tf.keras.layers.UpSampling3D(size=(2, 2, 2))(x)
    x = tf.keras.layers.Conv3D(64, (3, 3, 3), padding="same", activation="relu", kernel_regularizer=l2)(x)
    x = tf.keras.layers.Conv3D(32, (3, 3, 3), padding="same", activation="relu", kernel_regularizer=l2)(x)
    return x

def _upscale_5_to_20(shared_features: tf.Tensor, l2: tf.keras.regularizers.Regularizer) -> tf.Tensor:
    """Latent 5x5x5 -> 20x20x20 via two 2x upsampling stages + conv."""
    x = tf.keras.layers.Dense(5 * 5 * 5 * 32, activation="relu")(shared_features)
    x = tf.keras.layers.Reshape((5, 5, 5, 32))(x)
    x = tf.keras.layers.UpSampling3D(size=(2, 2, 2))(x)  # 5 -> 10
    x = tf.keras.layers.Conv3D(64, (3, 3, 3), padding="same", activation="relu", kernel_regularizer=l2)(x)
    x = tf.keras.layers.UpSampling3D(size=(2, 2, 2))(x)  # 10 -> 20
    x = tf.keras.layers.Conv3D(32, (3, 3, 3), padding="same", activation="relu", kernel_regularizer=l2)(x)
    return x

def _build_voxel_decoder(
    shared_features: tf.Tensor,
    grid_shape: Tuple[int, int, int],
    l2_reg: float,
) -> tf.Tensor:
    """
    Map backbone features to a coarse 3D occupancy volume.

    Supported decoder grids:
      - (10, 10, 10): 5^3 latent + 2x upsample
      - (20, 20, 20): 5^3 latent + staged upsample to 20^3
    """
    grid = tuple(int(v) for v in grid_shape)
    if grid == (10, 10, 10):
        x = _upscale_5_to_10(shared_features, l2=tf.keras.regularizers.l2(l2_reg))
    elif grid == (20, 20, 20):
        x = _upscale_5_to_20(shared_features, l2=tf.keras.regularizers.l2(l2_reg))
    else:
        raise ValueError(
            f"Unsupported voxel_grid={grid_shape}. Supported grids: (10,10,10) and (20,20,20)."
        )
    return tf.keras.layers.Conv3D(
        1, (1, 1, 1), activation="sigmoid", name="voxel_occupancy"
    )(x)

