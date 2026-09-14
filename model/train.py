
# PLACENet - Simplified CNN-based model with bounding box output and confidence output
# Supports smooth_l1, ciou, and hybrid loss types
# Supports greedy or JV (Jonker-Volgenant) matching

from __future__ import annotations
import gc
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import json
from sklearn.model_selection import KFold
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.utils import register_keras_serializable

from scipy.optimize import linear_sum_assignment

if TYPE_CHECKING:
    from preprocessing.load_data import PLACEDataset

from preprocessing.load_data import label_grid_shape, label_output_dim, label_target_dim

from .loss_bbox import BBoxMatchingLoss, PLACENetCustomLoss
from .loss_voxel import VoxelFocalLoss
from .backbone import _build_cnn_backbone, _build_transformer_backbone
from .head_bbox import _transform_box_params, _transform_position_params
from .head_voxel import _upscale_5_to_10, _upscale_5_to_20, _build_voxel_decoder


from .model_config import PLACENetConfig, PLACENetTrainingResult
from preprocessing.augmenter import PLACESpectrumAugmenter

# ============================================================================
# Model creation and training
# ============================================================================

class ValidationIoUMetric(tf.keras.callbacks.Callback):
    """
    Compute mean validation IoU after slot matching at the end of each epoch.
    Writes ``val_iou`` into ``logs`` for EarlyStopping / history (higher is better).
    """

    def __init__(
        self,
        x_val: np.ndarray,
        y_val: np.ndarray,
        *,
        confidence_threshold: float = 0.5,
        matching_strategy: str = "greedy",
        batch_size: int = 32,
        label_mode: str = "bbox",
    ):
        super().__init__()
        self.x_val = x_val
        self.y_val = np.asarray(y_val, dtype=np.float32)
        self.confidence_threshold = confidence_threshold
        self.matching_strategy = matching_strategy.lower()
        self.batch_size = batch_size
        self.label_mode = label_mode

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        from evaluation.evaluator import (
            apply_greedy_matching,
            apply_jv_matching,
            compute_iou_per_slot_stats,
        )

        logs = logs or {}
        y_pred = self.model.predict(self.x_val, batch_size=self.batch_size, verbose=0)
        if self.matching_strategy == "jv":
            y_pred_matched = apply_jv_matching(
                self.y_val,
                y_pred,
                confidence_threshold=self.confidence_threshold,
                label_mode=self.label_mode,
            )
        else:
            y_pred_matched = apply_greedy_matching(
                self.y_val,
                y_pred,
                confidence_threshold=self.confidence_threshold,
                label_mode=self.label_mode,
            )
        mean_iou, _, count = compute_iou_per_slot_stats(
            self.y_val, y_pred_matched, label_mode=self.label_mode
        )
        logs["val_iou"] = float(mean_iou) if count > 0 else 0.0


class ValidationPositionMetric(tf.keras.callbacks.Callback):
    """Mean matched L2 position error on validation set (lower is better)."""

    def __init__(
        self,
        x_val: np.ndarray,
        y_val: np.ndarray,
        *,
        confidence_threshold: float = 0.5,
        matching_strategy: str = "greedy",
        batch_size: int = 32,
        label_mode: str = "bbox",
    ):
        super().__init__()
        self.x_val = x_val
        self.y_val = np.asarray(y_val, dtype=np.float32)
        self.confidence_threshold = confidence_threshold
        self.matching_strategy = matching_strategy.lower()
        self.batch_size = batch_size
        self.label_mode = label_mode

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        from evaluation.evaluator import (
            apply_greedy_matching,
            apply_jv_matching,
            compute_position_mae_per_slot_stats,
        )

        logs = logs or {}
        y_pred = self.model.predict(self.x_val, batch_size=self.batch_size, verbose=0)
        if self.matching_strategy == "jv":
            y_pred_matched = apply_jv_matching(
                self.y_val,
                y_pred,
                confidence_threshold=self.confidence_threshold,
                label_mode=self.label_mode,
            )
        else:
            y_pred_matched = apply_greedy_matching(
                self.y_val,
                y_pred,
                confidence_threshold=self.confidence_threshold,
                label_mode=self.label_mode,
            )
        mean_mae, _, count = compute_position_mae_per_slot_stats(
            self.y_val, y_pred_matched, label_mode=self.label_mode
        )
        logs["val_position_mae"] = float(mean_mae) if count > 0 else 0.0


class ValidationVoxelIoUMetric(tf.keras.callbacks.Callback):
    """Mean IoU over coarse occupancy voxels on the validation set."""

    def __init__(
        self,
        x_val: np.ndarray,
        y_val: np.ndarray,
        *,
        voxel_mask: Optional[np.ndarray] = None,
        threshold: float = 0.5,
        batch_size: int = 32,
    ):
        super().__init__()
        self.x_val = x_val
        self.y_val = np.asarray(y_val, dtype=np.float32)
        self.threshold = threshold
        self.batch_size = batch_size
        self.voxel_mask = None if voxel_mask is None else np.asarray(voxel_mask, dtype=np.float32)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        logs = logs or {}
        y_pred = self.model.predict(self.x_val, batch_size=self.batch_size, verbose=0)
        y_true = self.y_val
        if y_true.ndim == 4:
            y_true = y_true[..., np.newaxis]
        pred_bin = (y_pred >= self.threshold).astype(np.float32)
        true_bin = (y_true >= 0.5).astype(np.float32)
        if self.voxel_mask is not None:
            mask = self.voxel_mask.reshape(1, *self.voxel_mask.shape, 1)
            pred_bin = pred_bin * mask
            true_bin = true_bin * mask
        inter = np.sum(pred_bin * true_bin, axis=(1, 2, 3, 4))
        union = np.sum((pred_bin + true_bin > 0).astype(np.float32), axis=(1, 2, 3, 4))
        iou = inter / (union + 1e-7)
        logs["val_voxel_iou"] = float(np.mean(iou))


class PLACENetCore:
    """Core model creation and training logic."""
    
    def __init__(self, train_labels: np.ndarray):
        pass
        self.train_labels = train_labels

    @staticmethod
    def add_confidence_to_labels(
        labels: np.ndarray,
        label_mode: str = "bbox",
    ) -> np.ndarray:
        """
        Add confidence dimension to labels for YOLO-style training.

        bbox mode: (N, S, 6) -> (N, S, 7)
        position mode: (N, S, 3) -> (N, S, 4)
        """
        labels = np.asarray(labels, dtype=np.float32)
        target_dim = label_target_dim(label_mode)
        output_dim = label_output_dim(label_mode)
        if labels.shape[-1] != target_dim:
            raise ValueError(
                f"Expected labels with last dim {target_dim} for label_mode={label_mode!r}, "
                f"got shape {labels.shape}"
            )
        N, S, _ = labels.shape
        labels_with_conf = np.zeros((N, S, output_dim), dtype=np.float32)
        labels_with_conf[:, :, :target_dim] = labels
        valid = ~np.all(np.isclose(labels, 0.0, atol=1e-3), axis=-1)
        labels_with_conf[:, :, target_dim] = valid.astype(np.float32)
        return labels_with_conf

    def _infer_center_bounds(
        self,
        train_labels: np.ndarray,
        label_mode: str = "bbox",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Infer center coordinate bounds from valid training labels.

        Returns:
            (center_min, center_max), each shape (3,) for x/y/z centre.
        """
        labels_arr = np.asarray(train_labels, dtype=np.float32)
        target_dim = label_target_dim(label_mode)
        if labels_arr.shape[-1] >= target_dim + 1:
            targets = labels_arr[..., :target_dim]
        elif labels_arr.shape[-1] == target_dim:
            targets = labels_arr
        else:
            raise ValueError(
                f"Expected train labels with {target_dim} or {target_dim + 1} dims for "
                f"label_mode={label_mode!r}, got shape {labels_arr.shape}"
            )

        valid_mask = ~np.all(np.isclose(targets, 0.0, atol=1e-3), axis=-1)
        if not np.any(valid_mask):
            return np.array([0.0, 0.0, 0.0], dtype=np.float32), np.array([100.0, 100.0, 100.0], dtype=np.float32)

        if label_mode == "position":
            centers_valid = targets[valid_mask]
        else:
            boxes = targets
            centers = np.stack([boxes[..., 1], boxes[..., 3], boxes[..., 5]], axis=-1)
            centers_valid = centers[valid_mask]

        center_min = centers_valid.min(axis=0).astype(np.float32)
        center_max = centers_valid.max(axis=0).astype(np.float32)
        margin = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        center_min = center_min - margin
        center_max = center_max + margin
        center_max = np.maximum(center_max, center_min + 1e-3)
        return center_min, center_max

    def make_CNN_model(
        self,
        train_labels: np.ndarray,
        config: PLACENetConfig,
        input_shape: Optional[Tuple[int, ...]] = None,
    ):
        """
        Create CNN model with YOLO-style output (targets + confidence).

        bbox mode output: (max_sources, 7) = [boxes(6), confidence(1)]
        position mode output: (max_sources, 4) = [xyz(3), confidence(1)]
        """
        if config.label_mode == "voxel":
            return self.make_voxel_model(config, input_shape=input_shape)

        if input_shape is None:
            input_shape = (16, 152, 1)  # Default shape

        target_dim = label_target_dim(config.label_mode)
        output_dim = label_output_dim(config.label_mode)
        
        input_layer = tf.keras.layers.Input(shape=input_shape)

        backbone = config.backbone.lower()
        if backbone == "cnn":
            shared_features = _build_cnn_backbone(input_layer, config)
        elif backbone == "transformer":
            shared_features = _build_transformer_backbone(
                input_layer, config, input_shape
            )
        else:
            raise ValueError(
                f"config.backbone must be 'cnn' or 'transformer', got {config.backbone!r}"
            )
        
        center_min, center_max = self._infer_center_bounds(train_labels, config.label_mode)
        center_min_list = center_min.tolist()
        center_scale_list = (center_max - center_min).tolist()

        if config.label_mode == "position":
            target_raw = tf.keras.layers.Dense(
                config.max_sources * target_dim,
                activation="linear",
                kernel_regularizer=tf.keras.regularizers.l2(config.l2_reg),
            )(shared_features)
            target_raw = tf.keras.layers.Reshape((config.max_sources, target_dim))(target_raw)
            target_output = tf.keras.layers.Lambda(
                _transform_position_params,
                arguments={"center_min": center_min_list, "center_scale": center_scale_list},
            )(target_raw)
        else:
            target_raw = tf.keras.layers.Dense(
                config.max_sources * target_dim,
                activation="linear",
                kernel_regularizer=tf.keras.regularizers.l2(config.l2_reg),
            )(shared_features)
            target_raw = tf.keras.layers.Reshape((config.max_sources, target_dim))(target_raw)
            target_output = tf.keras.layers.Lambda(
                _transform_box_params,
                arguments={"center_min": center_min_list, "center_scale": center_scale_list},
            )(target_raw)
        
        conf_output = tf.keras.layers.Dense(
            config.max_sources, activation="sigmoid",
            kernel_regularizer=tf.keras.regularizers.l2(config.l2_reg)
        )(shared_features)
        conf_output = tf.keras.layers.Reshape((config.max_sources, 1))(conf_output)
        
        output = tf.keras.layers.Concatenate(axis=-1)([target_output, conf_output])
        
        model = tf.keras.Model(inputs=input_layer, outputs=output)

        loss_fn = BBoxMatchingLoss(
            base=config.loss_type,
            max_sources=config.max_sources,
            delta=config.delta,
            smooth_weight=config.smooth_weight,
            ciou_weight=config.ciou_weight,
            confidence_weight=config.confidence_weight,
            noobj_weight=config.noobj_weight,
            label_mode=config.label_mode,
            position_axis_weights=config.position_axis_weights,
            init_log_var_box=getattr(config, "init_log_var_box", 0.0),
            init_log_var_conf=getattr(config, "init_log_var_conf", 0.0),
        )
        ## Update: homoscedastic uncertainty weighting
        model._trainable_variables.extend([loss_fn.s_box, loss_fn.s_conf])
        
        print(
            f"[make_CNN_model] backbone={backbone}, label_mode={config.label_mode}, "
            f"matching={config.matching_strategy}, base={config.loss_type}, "
            f"input_shape={input_shape}, output_dim={output_dim}"
        )

        opt = tf.keras.optimizers.Adam(learning_rate=config.learning_rate, clipnorm=1.0)
        model.compile(optimizer=opt, loss=loss_fn, metrics=[], jit_compile=False)
        return model

    def make_voxel_model(
        self,
        config: PLACENetConfig,
        input_shape: Optional[Tuple[int, ...]] = None,
    ):
        """Backbone + occupancy head (no matching, no confidence slots)."""
        if input_shape is None:
            input_shape = (16, 152, 1)
        input_layer = tf.keras.layers.Input(shape=input_shape)
        backbone = config.backbone.lower()
        if backbone == "cnn":
            shared_features = _build_cnn_backbone(input_layer, config)
        elif backbone == "transformer":
            shared_features = _build_transformer_backbone(input_layer, config, input_shape)
        else:
            raise ValueError(
                f"config.backbone must be 'cnn' or 'transformer', got {config.backbone!r}"
            )
        output = _build_voxel_decoder(shared_features, config.voxel_grid, config.l2_reg)
        model = tf.keras.Model(inputs=input_layer, outputs=output)

        voxel_mask = None
        if config.voxel_mask_loss:
            if config.voxel_mask is not None:
                voxel_mask = config.voxel_mask
            else:
                voxel_mask = np.ones(config.voxel_grid, dtype=np.float32)
        loss_fn = VoxelFocalLoss(
            gamma=config.voxel_focal_gamma,
            alpha=config.voxel_focal_alpha,
            voxel_mask=voxel_mask,
        )
        print(
            f"[make_voxel_model] backbone={backbone}, grid={config.voxel_grid}, "
            f"cells={int(np.prod(config.voxel_grid))}, masked_loss={config.voxel_mask_loss}, "
            f"input_shape={input_shape}"
        )
        opt = tf.keras.optimizers.Adam(learning_rate=config.learning_rate, clipnorm=1.0)
        model.compile(optimizer=opt, loss=loss_fn, metrics=[], jit_compile=False)
        return model

    @staticmethod
    def _build_augmenter(config: PLACENetConfig) -> Optional[PLACESpectrumAugmenter]:
        """Create an augmenter from config, or None when disabled."""
        if not config.enable_augmentation or config.augmentation_multiplier <= 0:
            return None
        kwargs = {
            "poisson_noise": True,
            "energy_shift": True,
            "intensity_scale": False,
            "detector_dropout": True,
            "energy_shift_range": 0.02,
            "intensity_scale_range": (0.8, 1.2),
            "detector_dropout_prob": 0.1,
            "poisson_scale": 500.0,
        }
        kwargs.update(config.augmenter_kwargs)
        return PLACESpectrumAugmenter(**kwargs)

    @staticmethod
    def build_training_callbacks(
        config: PLACENetConfig,
        x_val: np.ndarray,
        y_val: np.ndarray,
    ) -> List[Any]:
        """Early stopping (+ val IoU or val position MAE when monitoring those metrics)."""
        monitor = config.early_stopping_monitor
        callbacks: List[Any] = []

        if config.label_mode == "position" and monitor == "val_iou":
            print(
                "[build_training_callbacks] label_mode=position: remapping early_stopping_monitor "
                "from 'val_iou' to 'val_position_mae'"
            )
            monitor = "val_position_mae"
        if config.label_mode == "voxel" and monitor in ("val_iou", "val_position_mae"):
            print(
                f"[build_training_callbacks] label_mode=voxel: remapping early_stopping_monitor "
                f"from {monitor!r} to 'val_voxel_iou'"
            )
            monitor = "val_voxel_iou"

        if monitor == "val_voxel_iou":
            if config.voxel_mask is not None:
                voxel_mask = config.voxel_mask
            else:
                voxel_mask = np.ones(config.voxel_grid, dtype=np.float32)
            callbacks.append(
                ValidationVoxelIoUMetric(
                    x_val,
                    y_val,
                    voxel_mask=voxel_mask,
                    batch_size=config.batch_size,
                )
            )
            stop_mode = "max"
        elif monitor == "val_iou":
            callbacks.append(
                ValidationIoUMetric(
                    x_val,
                    y_val,
                    confidence_threshold=config.val_iou_confidence_threshold,
                    batch_size=config.batch_size,
                    label_mode=config.label_mode,
                )
            )
            stop_mode = "max"
        elif monitor == "val_position_mae":
            callbacks.append(
                ValidationPositionMetric(
                    x_val,
                    y_val,
                    confidence_threshold=config.val_iou_confidence_threshold,
                    batch_size=config.batch_size,
                    label_mode=config.label_mode,
                )
            )
            stop_mode = "min"
        elif monitor == "val_loss":
            stop_mode = "min"
        else:
            raise ValueError(
                "early_stopping_monitor must be 'val_iou', 'val_position_mae', "
                "'val_voxel_iou', or 'val_loss', "
                f"got {config.early_stopping_monitor!r}"
            )

        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor=monitor,
                mode=stop_mode,
                patience=config.early_stopping_patience,
                restore_best_weights=True,
                verbose=1,
            )
        )
        
        if config.warmup_epochs > 0:
            def lr_scheduler(epoch, lr):
                if epoch < config.warmup_epochs:
                    return config.learning_rate * ((epoch + 1) / config.warmup_epochs)
                return config.learning_rate
            callbacks.append(tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=0))

        return callbacks

    @staticmethod
    def _augment_train_fold(
        train_data: np.ndarray,
        train_labels: np.ndarray,
        config: PLACENetConfig,
        augmenter: Optional[PLACESpectrumAugmenter],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Augment train-fold only to avoid train/validation leakage."""
        if augmenter is None:
            return train_data, train_labels

        augmented_batches = [train_data]
        augmented_labels = [train_labels]
        original_size = len(train_data)
        for _ in range(config.augmentation_multiplier):
            augmented_batches.append(augmenter.augment_batch(train_data))
            augmented_labels.append(train_labels)

        train_data_aug = np.concatenate(augmented_batches, axis=0)
        train_labels_aug = np.concatenate(augmented_labels, axis=0)
        permutation = np.random.permutation(len(train_data_aug))
        train_data_aug = train_data_aug[permutation]
        train_labels_aug = train_labels_aug[permutation]

        print(
            f"Augmented train-fold size: {len(train_data_aug)} samples "
            f"(from {original_size}, multiplier {config.augmentation_multiplier + 1}x)"
        )
        return train_data_aug, train_labels_aug

    def do_kfold(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        config: PLACENetConfig,
        label: str,
    ) -> Tuple[List[Dict[str, float]], List[Any]]:
        """
        Perform k-fold cross-validation training.
        
        Args:
            data: Input data (N, H, W, C)
            labels: Labels (N, max_sources, target_dim) - converted to target_dim+1 with confidence
            config: PLACENetConfig instance
            label: Run label for file naming
            
        Returns:
            metrics_summary: List of metric dictionaries per fold
            history_list: List of training histories per fold
        """
        history_list = []
        metrics_summary = []
        file_label = config.run_name if config.run_name else label
        results_dir = Path(f"Results_{file_label}")
        results_dir.mkdir(parents=True, exist_ok=True)
        labels = np.asarray(labels, dtype=np.float32)
        if config.label_mode == "voxel":
            nx, ny, nz = config.voxel_grid
            if labels.ndim != 4 or labels.shape[1:] != (nx, ny, nz):
                raise ValueError(
                    f"Label shape mismatch for label_mode='voxel': expected "
                    f"(N, {nx}, {ny}, {nz}), got {labels.shape}. "
                    "Ensure PLACEPrepConfig.label_mode and voxel_grid match PLACENetConfig."
                )
            labels = labels[..., np.newaxis]
        else:
            expected_dim = label_target_dim(config.label_mode)
            if labels.shape[-1] != expected_dim:
                raise ValueError(
                    f"Label shape mismatch for label_mode={config.label_mode!r}: expected last dim "
                    f"{expected_dim}, got shape {labels.shape}. "
                    "Ensure PLACEPrepConfig.label_mode matches PLACENetConfig.label_mode."
                )
        augmenter = self._build_augmenter(config)

        run_config = {
            "label_mode": config.label_mode,
            "max_sources": config.max_sources,
            "loss_type": config.loss_type,
            "matching_strategy": config.matching_strategy,
            "backbone": config.backbone,
            "position_axis_weights": list(config.position_axis_weights),
        }
        if config.label_mode == "voxel":
            run_config["voxel_grid"] = list(config.voxel_grid)
            run_config["voxel_mask_loss"] = config.voxel_mask_loss
        with open(results_dir / f"run_config_{file_label}.json", "w") as f:
            json.dump(run_config, f, indent=2)

        for kfold, (train, test) in enumerate(KFold(n_splits=config.folds, shuffle=True, random_state=42).split(data, labels)):
            tf.keras.backend.clear_session()
            gc.collect()
            try:
                tf.config.experimental.reset_memory_stats("GPU:0")
            except Exception:
                pass

            # Set random seeds AFTER clear_session() to ensure reproducible weight initialization
            np.random.seed(42) # Set the random seed to 42 to compare when different models are trained on the same data
            tf.random.set_seed(42) # Set the random seed to 42 to compare when different models are trained on the same data

            train_data_fold = data[train]
            train_labels_fold = labels[train]
            train_data_fold, train_labels_fold = self._augment_train_fold(
                train_data_fold,
                train_labels_fold,
                config,
                augmenter,
            )

            # Prepare labels
            if config.label_mode == "voxel":
                train_labels_prep = train_labels_fold
                test_labels_prep = labels[test]
            else:
                train_labels_prep = self.add_confidence_to_labels(
                    train_labels_fold, label_mode=config.label_mode
                )
                test_labels_prep = self.add_confidence_to_labels(
                    labels[test], label_mode=config.label_mode
                )
            
            # Infer input shape from data
            input_shape = None
            if len(data) > 0:
                input_shape = data.shape[1:]  # (height, width, channels)
            
            # Create the model
            model = self.make_CNN_model(train_labels_prep, config, input_shape=input_shape)

            print(
                f"[{file_label}] Fold {kfold+1}/{config.folds}: "
                f"train={data[train].shape[0]} samples, val={data[test].shape[0]} samples, "
                f"early_stopping={config.early_stopping_monitor!r} (patience={config.early_stopping_patience})"
            )

            # Save the test data and labels for later evaluation
            np.savez(results_dir / f"data_labels_test_{file_label}_kf{kfold}.npz", data=data[test], labels=test_labels_prep)

            data_test_2d = data[test].reshape(data[test].shape[0], -1)
            labels_test_2d = test_labels_prep.reshape(test_labels_prep.shape[0], -1)
            np.savetxt(results_dir / f"data_test_{file_label}_kf{kfold}.csv", data_test_2d, delimiter=" ", fmt="%.8g")
            np.savetxt(results_dir / f"labels_test_{file_label}_kf{kfold}.csv", labels_test_2d, delimiter=" ", fmt="%.8g")

            log_dir = results_dir / "logs" / f"{file_label}_kf{kfold}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=str(log_dir), histogram_freq=0)
            training_callbacks = self.build_training_callbacks(
                config,
                data[test],
                test_labels_prep,
            )

            # Train the model and save the history
            history = model.fit(
                train_data_fold, train_labels_prep,
                epochs=config.epochs,
                batch_size=config.batch_size,
                callbacks=[*training_callbacks, tensorboard_callback],
                validation_data=(data[test], test_labels_prep),
                verbose=0
            )

            # Save the metrics for this fold
            final_epoch = history.history
            fold_metrics = {metric: values[-1] for metric, values in final_epoch.items()}
            fold_metrics["fold"] = kfold
            metrics_summary.append(fold_metrics)
            history_list.append(history)

            # Save loss curve for this fold
            try:
                from plot.plot_loss_curves import save_loss_curves
                save_loss_curves(
                    [history],
                    results_dir / f"loss_curves_{file_label}_kf{kfold}.pdf"
                )
            except Exception as e:
                print(f"Warning: Could not save loss curve for fold {kfold}: {e}")

            model.save(results_dir / f"model_{file_label}_kf{kfold}.keras")

            del model, history
            gc.collect()
            tf.keras.backend.clear_session()
            try:
                tf.config.experimental.reset_memory_stats("GPU:0")
            except Exception:
                pass

        df = pd.DataFrame(metrics_summary) # Save the metrics to a CSV file
        df.to_csv(results_dir / f"metrics_{file_label}.csv", sep=" ", index=False)

        with open(results_dir / f"metrics_{file_label}.json", "w") as f: # Save the metrics to a JSON file
            json.dump(metrics_summary, f, indent=2)

        with open(results_dir / f"metrics_summary_{file_label}.txt", "w") as f:
            f.write(f"Metrics Summary for {file_label}\n")
            f.write("=" * 50 + "\n\n")
            f.write("Summary Statistics:\n")
            f.write(df.describe().to_string())
            f.write("\n\nRaw Data:\n")
            f.write(df.to_string())

        return metrics_summary, history_list


# ============================================================================
# Main PLACENet class
# ============================================================================

class PLACENet:
    """Simplified PLACENet model with CNN architecture and YOLO-style output."""
    
    def __init__(self, config: PLACENetConfig):
        self.config = config

    def train_concatenated(
        self,
        datasets: Sequence["PLACEDataset"],
        run_label: Optional[str] = None
    ) -> PLACENetTrainingResult:
        """Train on concatenated datasets."""
        data, labels = self._concatenate(datasets)
        label = run_label or self.config.run_name or "run"
        metrics, histories = self._run_training(data, labels, label, run_name=self.config.run_name or label)
        return PLACENetTrainingResult(label, metrics, histories)

    def train_per_dataset(
        self,
        datasets: Sequence["PLACEDataset"],
        run_labels: Optional[Sequence[str]] = None
    ) -> List[PLACENetTrainingResult]:
        """Train on each dataset separately."""
        results: List[PLACENetTrainingResult] = []
        for idx, dataset in enumerate(datasets):
            label = (run_labels[idx % len(run_labels)] if run_labels else dataset.name)
            data, labels = self._align(dataset.data, dataset.labels)
            metrics, histories = self._run_training(data, labels, label, run_name=self.config.run_name or label)
            results.append(PLACENetTrainingResult(
                run_label=label,
                metrics=metrics,
                histories=histories,
                dataset_name=dataset.name
            ))
        return results

    def _run_training(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        run_label: str,
        run_name: Optional[str] = None,
    ) -> Tuple[List[Dict[str, float]], List[Any]]:
        """Run k-fold training."""
        runner = PLACENetCore(labels)#, self.config.pad_value)
        return runner.do_kfold(data, labels, self.config, run_label)

    def _concatenate(
        self,
        datasets: Sequence["PLACEDataset"]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Concatenate multiple datasets."""
        aligned = [self._align(ds.data, ds.labels) for ds in datasets]
        data_parts = [item[0] for item in aligned if len(item[0])]
        label_parts = [item[1] for item in aligned if len(item[1])]
        if not data_parts or not label_parts:
            raise ValueError("No datasets available for concatenated training")
        return np.concatenate(data_parts), np.concatenate(label_parts)

    @staticmethod
    def _align(data: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Align data and labels to same length."""
        if len(data) == len(labels):
            return data, labels
        length = min(len(data), len(labels))
        return data[:length], labels[:length]

