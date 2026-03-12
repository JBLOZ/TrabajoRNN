from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import tensorflow as tf


def oversample_indices_per_class(y: np.ndarray, random_state: int = 42) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    y = np.asarray(y).astype(int)
    class_to_indices: dict[int, np.ndarray] = {}
    max_count = 0
    for cls in np.unique(y):
        indices = np.where(y == cls)[0]
        class_to_indices[int(cls)] = indices
        max_count = max(max_count, len(indices))

    sampled = []
    for cls, indices in class_to_indices.items():
        sampled.extend(rng.choice(indices, size=max_count, replace=True).tolist())
    sampled = np.asarray(sampled, dtype=np.int64)
    rng.shuffle(sampled)
    return sampled


class BalancedBatchSequence(tf.keras.utils.Sequence):
    def __init__(
        self,
        inputs: np.ndarray | dict[str, np.ndarray],
        y_one_hot: np.ndarray,
        batch_size: int = 128,
        steps_per_epoch: int = 200,
        random_state: int = 42,
    ) -> None:
        self.inputs = inputs
        self.y_one_hot = np.asarray(y_one_hot, dtype=np.float32)
        self.y = np.argmax(self.y_one_hot, axis=1).astype(int)
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.rng = np.random.default_rng(random_state)

        self.class_indices: dict[int, np.ndarray] = {}
        for cls in np.unique(self.y):
            self.class_indices[int(cls)] = np.where(self.y == cls)[0]

        self.classes = sorted(self.class_indices.keys())
        self.per_class = max(batch_size // max(len(self.classes), 1), 1)

    def __len__(self) -> int:
        return self.steps_per_epoch

    def _slice_inputs(self, indices: np.ndarray) -> Any:
        if isinstance(self.inputs, dict):
            return {k: v[indices] for k, v in self.inputs.items()}
        return self.inputs[indices]

    def __getitem__(self, idx: int) -> tuple[Any, np.ndarray]:
        batch_indices = []
        for cls in self.classes:
            choices = self.rng.choice(self.class_indices[cls], size=self.per_class, replace=True)
            batch_indices.extend(choices.tolist())
        batch_indices = np.asarray(batch_indices[: self.batch_size], dtype=np.int64)
        self.rng.shuffle(batch_indices)
        return self._slice_inputs(batch_indices), self.y_one_hot[batch_indices]
