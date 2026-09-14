import json
import tensorflow as tf
from pathlib import Path
import glob
import numpy as np

def robust_load_model(model_path, res_dir, labels_test, data_test):
    """
    Attempts to load a Keras model directly. If that fails (due to custom layers/losses),
    it rebuilds the model architecture from the config in `res_dir` and loads the weights.
    """
    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        print(f"Standard load_model failed ({e}), rebuilding architecture and loading weights...")
        from model.model_config import PLACENetConfig
        from model.train import PLACENetCore
            
        run_config_paths = glob.glob(str(Path(res_dir) / "run_config*.json"))
        config_dict = {}
        if run_config_paths:
            with open(run_config_paths[0]) as f:
                config_dict = json.load(f)
                
        pad_value = config_dict.get("pad_value", -1)
        max_sources = config_dict.get("max_sources", 3)
        last_dim = labels_test.shape[-1]
        
        # We can infer label_mode from the config or from dim
        label_mode = config_dict.get("label_mode")
        if label_mode is None:
            # Fallback simple inference
            label_mode = "voxel" if last_dim == 1 else "bbox"
            
        # Reconstruct the dummy config
        dummy_config = PLACENetConfig(
            max_sources=max_sources,
            label_mode=label_mode,
            backbone=config_dict.get("backbone", "cnn")
        )
        
        # The voxel grid/mask might be needed if it was a voxel model, 
        # but for rebuilding architecture, only the shape matters which is handled by labels_test.
        if label_mode == "voxel":
            dummy_config.voxel_grid = tuple(config_dict.get("voxel_grid", (10, 10, 10)))
            
        core = PLACENetCore(np.array([]))
        
        if label_mode == "voxel":
            model = core.make_voxel_model(labels_test, dummy_config, input_shape=data_test.shape[1:])
        else:
            model = core.make_CNN_model(labels_test, dummy_config, input_shape=data_test.shape[1:])
            
        # Load weights into the rebuilt architecture
        model.load_weights(model_path, skip_mismatch=True)
        return model
