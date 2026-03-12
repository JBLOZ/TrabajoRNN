from __future__ import annotations

from typing import Any

import numpy as np
import tensorflow as tf


def temporal_occlusion_importance(
    model: tf.keras.Model,
    inputs: np.ndarray | dict[str, np.ndarray],
    class_index: int | None = None,
    baseline_value: float = 0.0,
    window: int = 1,
):
    """
    Occlude timesteps and measure probability drop for the target class.
    Supports a batch size of 1 for interpretability examples.
    """
    if isinstance(inputs, dict):
        # use the first key as reference timeline and occlude each branch independently
        result = {}
        base_pred = model({k: tf.convert_to_tensor(v, dtype=tf.float32) for k, v in inputs.items()}, training=False).numpy()
        if class_index is None:
            class_index = int(np.argmax(base_pred[0]))
        base_score = float(base_pred[0, class_index])
        for key, array in inputs.items():
            arr = np.asarray(array, dtype=np.float32).copy()
            importance = np.zeros(arr.shape[1], dtype=np.float32)
            for t in range(arr.shape[1]):
                occluded = arr.copy()
                occluded[:, t : t + window, :] = baseline_value
                new_inputs = {k: np.asarray(v, dtype=np.float32).copy() for k, v in inputs.items()}
                new_inputs[key] = occluded
                pred = model({k: tf.convert_to_tensor(v, dtype=tf.float32) for k, v in new_inputs.items()}, training=False).numpy()
                importance[t] = max(base_score - float(pred[0, class_index]), 0.0)
            result[key] = importance
        return result

    arr = np.asarray(inputs, dtype=np.float32).copy()
    base_pred = model(tf.convert_to_tensor(arr, dtype=tf.float32), training=False).numpy()
    if class_index is None:
        class_index = int(np.argmax(base_pred[0]))
    base_score = float(base_pred[0, class_index])

    importance = np.zeros(arr.shape[1], dtype=np.float32)
    for t in range(arr.shape[1]):
        occluded = arr.copy()
        occluded[:, t : t + window, :] = baseline_value
        pred = model(tf.convert_to_tensor(occluded, dtype=tf.float32), training=False).numpy()
        importance[t] = max(base_score - float(pred[0, class_index]), 0.0)
    return importance
