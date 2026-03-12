from __future__ import annotations

from typing import Sequence

import tensorflow as tf


def categorical_focal_loss(
    gamma: float = 2.0,
    alpha: Sequence[float] | None = None,
) -> tf.keras.losses.Loss:
    alpha_tensor = None
    if alpha is not None:
        alpha_tensor = tf.constant(alpha, dtype=tf.float32)

    def loss_fn(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce = -y_true * tf.math.log(y_pred)
        weight = tf.pow(1.0 - y_pred, gamma)
        if alpha_tensor is not None:
            ce = ce * alpha_tensor
        focal = weight * ce
        return tf.reduce_sum(focal, axis=-1)

    return loss_fn
