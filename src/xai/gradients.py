from __future__ import annotations

from typing import Any

import numpy as np
import tensorflow as tf


def _ensure_tensor_inputs(inputs: np.ndarray | dict[str, np.ndarray]) -> Any:
    if isinstance(inputs, dict):
        return {k: tf.convert_to_tensor(v, dtype=tf.float32) for k, v in inputs.items()}
    return tf.convert_to_tensor(inputs, dtype=tf.float32)


def _zero_baseline(inputs: np.ndarray | dict[str, np.ndarray]) -> np.ndarray | dict[str, np.ndarray]:
    if isinstance(inputs, dict):
        return {k: np.zeros_like(v, dtype=np.float32) for k, v in inputs.items()}
    return np.zeros_like(inputs, dtype=np.float32)


def _gather_target_score(preds: tf.Tensor, class_index: int | None) -> tf.Tensor:
    if class_index is None:
        class_index = int(tf.argmax(preds[0]).numpy())
    return preds[:, class_index]


def saliency_map(model: tf.keras.Model, inputs: np.ndarray | dict[str, np.ndarray], class_index: int | None = None):
    tensor_inputs = _ensure_tensor_inputs(inputs)
    with tf.GradientTape() as tape:
        if isinstance(tensor_inputs, dict):
            for value in tensor_inputs.values():
                tape.watch(value)
            preds = model(tensor_inputs, training=False)
        else:
            tape.watch(tensor_inputs)
            preds = model(tensor_inputs, training=False)
        target_score = _gather_target_score(preds, class_index)
    grads = tape.gradient(target_score, tensor_inputs)
    if isinstance(grads, dict):
        return {k: np.asarray(v.numpy()) for k, v in grads.items()}
    return np.asarray(grads.numpy())


def integrated_gradients(
    model: tf.keras.Model,
    inputs: np.ndarray | dict[str, np.ndarray],
    baseline: np.ndarray | dict[str, np.ndarray] | None = None,
    steps: int = 32,
    class_index: int | None = None,
):
    baseline = _zero_baseline(inputs) if baseline is None else baseline

    def _integrated(single_input, single_base):
        single_input = tf.convert_to_tensor(single_input, dtype=tf.float32)
        single_base = tf.convert_to_tensor(single_base, dtype=tf.float32)
        alphas = tf.linspace(0.0, 1.0, steps + 1)
        total = tf.zeros_like(single_input, dtype=tf.float32)
        for alpha in alphas:
            x = single_base + alpha * (single_input - single_base)
            with tf.GradientTape() as tape:
                tape.watch(x)
                preds = model(x, training=False)
                target_score = _gather_target_score(preds, class_index)
            grads = tape.gradient(target_score, x)
            total += grads
        avg_grads = total / tf.cast(len(alphas), tf.float32)
        return ((single_input - single_base) * avg_grads).numpy()

    if isinstance(inputs, dict):
        result = {}
        for k in inputs:
            def partial_model(x, training=False, _k=k):
                tmp = {kk: tf.convert_to_tensor(v, dtype=tf.float32) for kk, v in inputs.items()}
                tmp[_k] = x
                return model(tmp, training=training)
            # local wrapper
            class Wrapper(tf.keras.Model):
                def call(self, x, training=False):
                    return partial_model(x, training=training)
            wrapper = Wrapper()
            result[k] = _integrated(inputs[k], baseline[k])
        return result
    return _integrated(inputs, baseline)
