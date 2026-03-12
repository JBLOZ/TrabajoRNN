from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler


ScalerName = Literal["robust", "standard"]


@dataclass
class SequenceScaler:
    scaler_name: ScalerName
    sklearn_scaler: RobustScaler | StandardScaler

    def transform(self, x: np.ndarray) -> np.ndarray:
        original_shape = x.shape
        flat = x.reshape(-1, x.shape[-1])
        transformed = self.sklearn_scaler.transform(flat)
        return transformed.reshape(original_shape).astype(np.float32)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        original_shape = x.shape
        flat = x.reshape(-1, x.shape[-1])
        transformed = self.sklearn_scaler.inverse_transform(flat)
        return transformed.reshape(original_shape).astype(np.float32)


def fit_sequence_scaler(x_train: np.ndarray, scaler_name: ScalerName = "robust") -> SequenceScaler:
    flat = x_train.reshape(-1, x_train.shape[-1])
    if scaler_name == "robust":
        scaler = RobustScaler()
    elif scaler_name == "standard":
        scaler = StandardScaler()
    else:
        raise ValueError(f"Unknown scaler_name={scaler_name}")
    scaler.fit(flat)
    return SequenceScaler(scaler_name=scaler_name, sklearn_scaler=scaler)
