
import tensorflow as tf
from typing import Tuple
from .model_config import PLACENetConfig

def _build_cnn_backbone(
    input_layer: tf.Tensor, config: PLACENetConfig
) -> tf.Tensor:
    """2D CNN on (detectors, bins) — original green box."""
    l2 = tf.keras.regularizers.l2(config.l2_reg)
    x = tf.keras.layers.Conv2D(
        128, (3, 3), activation="relu", padding="same", kernel_regularizer=l2
    )(input_layer)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(
        64, (3, 3), activation="relu", padding="same", kernel_regularizer=l2
    )(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=l2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    shared = tf.keras.layers.Dense(64, activation="softplus", kernel_regularizer=l2)(x)
    return tf.keras.layers.BatchNormalization()(shared)

def _build_transformer_backbone(
    input_layer: tf.Tensor,
    config: PLACENetConfig,
    input_shape: Tuple[int, ...],
) -> tf.Tensor:
    """
    Per-detector Conv1D on time/bins, then self-attention across detectors.
    Input (B, n_det, n_bins, 1) -> shared features (B, 64).
    """
    if len(input_shape) != 3:
        raise ValueError(f"Expected input_shape (n_det, n_bins, channels), got {input_shape}")
    n_det, n_bins, _ = input_shape
    l2 = tf.keras.regularizers.l2(config.l2_reg)
    f1, f2 = config.transformer_conv_filters

    # (B, n_det, n_bins, 1): TimeDistributed Conv1D runs along n_bins per detector
    x = tf.keras.layers.Reshape((n_det, n_bins, 1))(input_layer)

    def _td_conv(filters: int, kernel: int = 3) -> tf.keras.layers.Layer:
        return tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv1D(
                filters,
                kernel_size=kernel,
                activation="relu",
                padding="same",
                kernel_regularizer=l2,
            )
        )

    x = _td_conv(f1)(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling1D(pool_size=2))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(x)
    x = _td_conv(f2)(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling1D(pool_size=2))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)

    attn = tf.keras.layers.MultiHeadAttention(
        num_heads=config.transformer_num_heads,
        key_dim=config.transformer_key_dim,
    )(query=x, value=x, key=x)
    x = tf.keras.layers.Add()([x, attn])
    x = tf.keras.layers.LayerNormalization()(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=l2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    shared = tf.keras.layers.Dense(64, activation="softplus", kernel_regularizer=l2)(x)
    return tf.keras.layers.BatchNormalization()(shared)

