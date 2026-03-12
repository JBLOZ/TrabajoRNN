from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from src.training.balanced_sampler import BalancedBatchSequence, oversample_indices_per_class
from src.training.losses import categorical_focal_loss
from src.training.models import build_v1_baseline, build_v2_improved, build_v3_dual_branch


@dataclass
class TrainingConfig:
    model_version: str
    batch_size: int = 128
    epochs: int = 50
    learning_rate: float = 1e-3
    clipnorm: float = 1.0
    loss_name: str = "categorical_crossentropy"
    balance_strategy: str = "class_weights"  # class_weights|focal|oversample|balanced_batch|none
    patience: int = 10
    lr_patience: int = 4
    min_lr: float = 1e-5
    checkpoint_path: str | None = None
    random_state: int = 42
    model_kwargs: dict[str, Any] | None = None


def _to_categorical(y: np.ndarray, num_classes: int) -> np.ndarray:
    return tf.keras.utils.to_categorical(y.astype(int), num_classes=num_classes)


def _compute_class_weights(y: np.ndarray) -> dict[int, float]:
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def _select_loss(cfg: TrainingConfig, y_train: np.ndarray, num_classes: int):
    if cfg.balance_strategy == "focal" or cfg.loss_name == "focal":
        class_weights = _compute_class_weights(y_train)
        alpha = [class_weights.get(i, 1.0) for i in range(num_classes)]
        return categorical_focal_loss(gamma=2.0, alpha=alpha)
    return "categorical_crossentropy"


def _make_callbacks(cfg: TrainingConfig) -> list[tf.keras.callbacks.Callback]:
    callbacks: list[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=cfg.patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=cfg.lr_patience,
            min_lr=cfg.min_lr,
            verbose=1,
        ),
    ]
    if cfg.checkpoint_path:
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=cfg.checkpoint_path,
                monitor="val_loss",
                save_best_only=True,
                verbose=1,
            )
        )
    return callbacks


def build_model_for_version(
    model_version: str,
    input_shapes: Mapping[str, tuple[int, int]],
    num_classes: int,
    model_kwargs: dict[str, Any] | None = None,
) -> tf.keras.Model:
    model_kwargs = model_kwargs or {}
    if model_version == "v1":
        return build_v1_baseline(
            input_shape=input_shapes["sequence_input"],
            num_classes=num_classes,
            **model_kwargs,
        )
    if model_version == "v2":
        return build_v2_improved(
            input_shape=input_shapes["sequence_input"],
            num_classes=num_classes,
            **model_kwargs,
        )
    if model_version == "v3":
        return build_v3_dual_branch(
            hrv_shape=input_shapes["hrv_input"],
            morph_shape=input_shapes["morph_input"],
            num_classes=num_classes,
            **model_kwargs,
        )
    raise ValueError(f"Unknown model_version={model_version}")


def fit_model(
    x_train: np.ndarray | dict[str, np.ndarray],
    y_train: np.ndarray,
    x_val: np.ndarray | dict[str, np.ndarray],
    y_val: np.ndarray,
    cfg: TrainingConfig,
    input_shapes: Mapping[str, tuple[int, int]],
    num_classes: int,
) -> tuple[tf.keras.Model, tf.keras.callbacks.History]:
    model = build_model_for_version(
        model_version=cfg.model_version,
        input_shapes=input_shapes,
        num_classes=num_classes,
        model_kwargs=cfg.model_kwargs,
    )
    loss = _select_loss(cfg, y_train, num_classes=num_classes)
    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate, clipnorm=cfg.clipnorm)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=["accuracy"],
    )

    y_train_ohe = _to_categorical(y_train, num_classes)
    y_val_ohe = _to_categorical(y_val, num_classes)

    callbacks = _make_callbacks(cfg)

    class_weight = None
    fit_x = x_train
    fit_y = y_train_ohe
    fit_kwargs: dict[str, Any] = {}

    if cfg.balance_strategy == "class_weights":
        class_weight = _compute_class_weights(y_train)
    elif cfg.balance_strategy == "oversample":
        sampled_idx = oversample_indices_per_class(y_train, random_state=cfg.random_state)
        if isinstance(x_train, dict):
            fit_x = {k: v[sampled_idx] for k, v in x_train.items()}
        else:
            fit_x = x_train[sampled_idx]
        fit_y = y_train_ohe[sampled_idx]
    elif cfg.balance_strategy == "balanced_batch":
        fit_x = BalancedBatchSequence(
            inputs=x_train,
            y_one_hot=y_train_ohe,
            batch_size=cfg.batch_size,
            steps_per_epoch=max(len(y_train) // cfg.batch_size, 50),
            random_state=cfg.random_state,
        )
        fit_y = None
        fit_kwargs["steps_per_epoch"] = len(fit_x)

    history = model.fit(
        fit_x,
        fit_y,
        validation_data=(x_val, y_val_ohe),
        batch_size=None if cfg.balance_strategy == "balanced_batch" else cfg.batch_size,
        epochs=cfg.epochs,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
        **fit_kwargs,
    )
    return model, history
