from __future__ import annotations

import numpy as np


HRV_FEATURE_NAMES = [
    "rr",
    "delta_rr",
    "mean_rr",
    "std_rr",
    "rmssd",
    "pnn50",
    "range_rr",
    "cv_rr",
]


def compute_prefix_hrv_features(rr_history: np.ndarray) -> np.ndarray:
    """Compute causal HRV features for each step in a history window."""
    rr_history = np.asarray(rr_history, dtype=np.float32)
    features = []
    for t in range(len(rr_history)):
        sub = rr_history[: t + 1]
        rr_val = float(sub[-1])
        delta = float(sub[-1] - sub[-2]) if len(sub) > 1 else 0.0
        mean_rr = float(np.mean(sub))
        std_rr = float(np.std(sub)) if len(sub) > 1 else 0.0
        diffs = np.diff(sub)
        rmssd = float(np.sqrt(np.mean(diffs ** 2))) if len(diffs) > 0 else 0.0
        pnn50 = float(np.mean(np.abs(diffs) > 0.05)) if len(diffs) > 0 else 0.0
        range_rr = float(np.max(sub) - np.min(sub))
        cv_rr = float(std_rr / mean_rr) if mean_rr > 0 else 0.0
        features.append([rr_val, delta, mean_rr, std_rr, rmssd, pnn50, range_rr, cv_rr])
    return np.asarray(features, dtype=np.float32)
