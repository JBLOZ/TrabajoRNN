from __future__ import annotations

from dataclasses import asdict
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import tensorflow as tf

from src.evaluation.metrics import compute_global_metrics
from src.training.train import TrainingConfig, fit_model


def run_directed_search(
    candidate_cfgs: Iterable[TrainingConfig],
    x_train: np.ndarray | dict[str, np.ndarray],
    y_train: np.ndarray,
    x_val: np.ndarray | dict[str, np.ndarray],
    y_val: np.ndarray,
    input_shapes: Mapping[str, tuple[int, int]],
    num_classes: int,
) -> tuple[pd.DataFrame, tf.keras.Model, TrainingConfig]:
    rows: list[dict[str, Any]] = []
    best_score = -np.inf
    best_model = None
    best_cfg = None

    for idx, cfg in enumerate(candidate_cfgs, start=1):
        model, history = fit_model(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            cfg=cfg,
            input_shapes=input_shapes,
            num_classes=num_classes,
        )
        y_val_prob = model.predict(x_val, verbose=0)
        y_val_pred = np.argmax(y_val_prob, axis=1)
        metrics = compute_global_metrics(y_true=y_val, y_pred=y_val_pred, y_prob=y_val_prob)
        row = {**asdict(cfg), **metrics, "search_iteration": idx}
        rows.append(row)

        if metrics["macro_f1"] > best_score:
            best_score = metrics["macro_f1"]
            best_model = model
            best_cfg = cfg

    if best_model is None or best_cfg is None:
        raise RuntimeError("No se pudo entrenar ninguna configuración candidata.")

    return pd.DataFrame(rows).sort_values("macro_f1", ascending=False), best_model, best_cfg
