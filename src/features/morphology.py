from __future__ import annotations

import numpy as np

MORPH_FEATURE_NAMES = [
    "max_amp",
    "min_amp",
    "ptp_amp",
    "mean_amp",
    "std_amp",
    "rms_amp",
    "energy",
    "abs_area",
    "max_slope",
    "min_slope",
    "diff_energy",
    "pre_post_energy_ratio",
    "r_peak_rel_pos",
    "zero_crossing_rate",
]


def extract_morphology_features(beat: np.ndarray) -> np.ndarray:
    beat = np.asarray(beat, dtype=np.float32)
    diff = np.diff(beat, prepend=beat[0])
    r_pos = int(np.argmax(beat))
    pre = beat[: r_pos + 1]
    post = beat[r_pos:]
    zero_cross = np.count_nonzero(np.diff(np.signbit(beat)))
    features = np.array(
        [
            float(np.max(beat)),
            float(np.min(beat)),
            float(np.ptp(beat)),
            float(np.mean(beat)),
            float(np.std(beat)),
            float(np.sqrt(np.mean(beat ** 2))),
            float(np.sum(beat ** 2)),
            float(np.sum(np.abs(beat))),
            float(np.max(diff)),
            float(np.min(diff)),
            float(np.sum(diff ** 2)),
            float(np.sum(pre ** 2) / (np.sum(post ** 2) + 1e-8)),
            float(r_pos / max(len(beat), 1)),
            float(zero_cross / max(len(beat) - 1, 1)),
        ],
        dtype=np.float32,
    )
    return features
