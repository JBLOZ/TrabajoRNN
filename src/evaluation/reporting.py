from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def plot_training_history(history, title_prefix: str = ""):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history.get("loss", []), label="train")
    axes[0].plot(history.history.get("val_loss", []), label="val")
    axes[0].set_title(f"{title_prefix} Loss")
    axes[0].legend()

    axes[1].plot(history.history.get("accuracy", []), label="train")
    axes[1].plot(history.history.get("val_accuracy", []), label="val")
    axes[1].set_title(f"{title_prefix} Accuracy")
    axes[1].legend()
    plt.tight_layout()
    return fig


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, class_names: dict[int, str], normalize: bool = False):
    labels = [class_names[i] for i in sorted(class_names)]
    cm = confusion_matrix(y_true, y_pred, normalize="true" if normalize else None)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de confusión" + (" normalizada" if normalize else ""))
    plt.tight_layout()
    return fig


def measure_inference_time(
    model: tf.keras.Model,
    sample_inputs: np.ndarray | dict[str, np.ndarray],
    repeats: int = 20,
) -> dict[str, float]:
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        _ = model.predict(sample_inputs, verbose=0)
        times.append(time.perf_counter() - start)
    arr = np.asarray(times)
    return {"mean_seconds": float(arr.mean()), "std_seconds": float(arr.std())}
