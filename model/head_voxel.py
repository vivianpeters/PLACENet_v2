import tensorflow as tf
from typing import Tuple

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

def _upscale_5_5_7_to_20_20_28(shared_features: tf.Tensor, l2: tf.keras.regularizers.Regularizer) -> tf.Tensor:
    """Latent 5x5x7 -> 20x20x28 (cubic voxels: 22.9 x 22.9 x 22.82 mm)."""
    x = tf.keras.layers.Dense(5 * 5 * 7 * 32, activation="relu")(shared_features)
    x = tf.keras.layers.Reshape((5, 5, 7, 32))(x)
    x = tf.keras.layers.UpSampling3D(size=(2, 2, 2))(x)  # 5, 5, 7 -> 10, 10, 14
    x = tf.keras.layers.Conv3D(64, (3, 3, 3), padding="same", activation="relu", kernel_regularizer=l2)(x)
    x = tf.keras.layers.UpSampling3D(size=(2, 2, 2))(x)  # 10, 10, 14 -> 20, 20, 28
    x = tf.keras.layers.Conv3D(32, (3, 3, 3), padding="same", activation="relu", kernel_regularizer=l2)(x)
    return x

def _upscale_5_5_7_to_10_10_14(shared_features: tf.Tensor, l2: tf.keras.regularizers.Regularizer) -> tf.Tensor:
    """Latent 5x5x7 -> 10x10x14 (cubic voxels: 45.8 x 45.8 x 45.64 mm)."""
    x = tf.keras.layers.Dense(5 * 5 * 7 * 32, activation="relu")(shared_features)
    x = tf.keras.layers.Reshape((5, 5, 7, 32))(x)
    x = tf.keras.layers.UpSampling3D(size=(2, 2, 2))(x)  # 5, 5, 7 -> 10, 10, 14
    x = tf.keras.layers.Conv3D(64, (3, 3, 3), padding="same", activation="relu", kernel_regularizer=l2)(x)
    x = tf.keras.layers.Conv3D(32, (3, 3, 3), padding="same", activation="relu", kernel_regularizer=l2)(x)
    return x

def _upscale_5_5_5_to_20_20_30(shared_features: tf.Tensor, l2: tf.keras.regularizers.Regularizer) -> tf.Tensor:
    """Latent 5x5x5 -> 20x20x30 via asymmetric upsampling."""
    x = tf.keras.layers.Dense(5 * 5 * 5 * 32, activation="relu")(shared_features)
    x = tf.keras.layers.Reshape((5, 5, 5, 32))(x)
    x = tf.keras.layers.UpSampling3D(size=(2, 2, 3))(x)  # 5, 5, 5 -> 10, 10, 15
    x = tf.keras.layers.Conv3D(64, (3, 3, 3), padding="same", activation="relu", kernel_regularizer=l2)(x)
    x = tf.keras.layers.UpSampling3D(size=(2, 2, 2))(x)  # 10, 10, 15 -> 20, 20, 30
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
      - (20, 20, 28): Cubic voxels (22.90 x 22.90 x 22.82 mm) [Recommended]
      - (10, 10, 14): Coarse cubic voxels (45.80 x 45.80 x 45.64 mm)
      - (20, 20, 30): Grid (22.90 x 22.90 x 21.30 mm)
      - (20, 20, 20): Legacy grid (22.90 x 22.90 x 31.95 mm)
      - (10, 10, 10): Legacy coarse grid
    """
    grid = tuple(int(v) for v in grid_shape)
    l2 = tf.keras.regularizers.l2(l2_reg)
    if grid == (20, 20, 28):
        x = _upscale_5_5_7_to_20_20_28(shared_features, l2=l2)
    elif grid == (10, 10, 14):
        x = _upscale_5_5_7_to_10_10_14(shared_features, l2=l2)
    elif grid == (20, 20, 30):
        x = _upscale_5_5_5_to_20_20_30(shared_features, l2=l2)
    elif grid == (20, 20, 20):
        x = _upscale_5_to_20(shared_features, l2=l2)
    elif grid == (10, 10, 10):
        x = _upscale_5_to_10(shared_features, l2=l2)
    else:
        raise ValueError(
            f"Unsupported voxel_grid={grid_shape}. Supported grids: (20,20,28) [cubic], (10,10,14) [cubic], (20,20,30), (20,20,20), (10,10,10)."
        )
    return tf.keras.layers.Conv3D(
        1, (1, 1, 1), activation="sigmoid", name="voxel_occupancy"
    )(x)

