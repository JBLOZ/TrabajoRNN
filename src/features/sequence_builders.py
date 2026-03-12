from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.aami import AAMIConfig
from src.data.mitbih_loader import (
    extract_centered_beat,
    filter_valid_beats,
    list_record_ids,
    load_record,
)
from src.features.hrv import compute_prefix_hrv_features, HRV_FEATURE_NAMES
from src.features.morphology import extract_morphology_features, MORPH_FEATURE_NAMES


@dataclass
class SequenceBuildConfig:
    data_dir: str
    label_mode: str = "aami_5"
    n_steps: int = 15
    horizon: int = 1
    lead: int = 0
    beat_window_before: int = 90
    beat_window_after: int = 162
    exclude_paced_records: bool = False


def build_dual_branch_dataset(config: SequenceBuildConfig) -> dict[str, Any]:
    cfg = AAMIConfig(label_mode=config.label_mode)
    hrv_sequences: list[np.ndarray] = []
    morph_sequences: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[str] = []
    meta_rows: list[dict[str, Any]] = []

    record_ids = list_record_ids(config.data_dir)
    for record_id in record_ids:
        record = load_record(config.data_dir, record_id, lead=config.lead)
        beat_samples, beat_symbols = filter_valid_beats(
            record, label_mode=config.label_mode, exclude_paced_records=config.exclude_paced_records
        )
        if len(beat_samples) <= config.n_steps + config.horizon:
            continue

        rr_intervals = np.diff(beat_samples) / record.fs  # rr[k] belongs to beat (k+1)

        for label_idx in range(config.n_steps + config.horizon, len(beat_samples)):
            history_end = label_idx - config.horizon
            history_start = history_end - config.n_steps + 1

            if history_start < 1:
                continue

            rr_history = rr_intervals[history_start - 1 : history_end]
            if len(rr_history) != config.n_steps:
                continue

            morph_history = []
            beat_centers = beat_samples[history_start : history_end + 1]
            valid_window = True
            for center in beat_centers:
                beat = extract_centered_beat(
                    record.signal,
                    int(center),
                    window_before=config.beat_window_before,
                    window_after=config.beat_window_after,
                )
                if beat is None:
                    valid_window = False
                    break
                morph_history.append(extract_morphology_features(beat))
            if not valid_window:
                continue

            hrv_seq = compute_prefix_hrv_features(rr_history)
            morph_seq = np.asarray(morph_history, dtype=np.float32)

            label_symbol = beat_symbols[label_idx]
            if label_symbol not in cfg.valid_symbols:
                continue
            label = cfg.mapping[label_symbol]

            hrv_sequences.append(hrv_seq)
            morph_sequences.append(morph_seq)
            labels.append(label)
            groups.append(record_id)
            meta_rows.append(
                {
                    "record_id": record_id,
                    "label_index": int(label_idx),
                    "label_symbol": str(label_symbol),
                    "label_class": int(label),
                    "label_sample": int(beat_samples[label_idx]),
                    "history_start_index": int(history_start),
                    "history_end_index": int(history_end),
                    "history_samples": [int(v) for v in beat_centers.tolist()],
                    "rr_history_seconds": [float(v) for v in rr_history.tolist()],
                }
            )

    if not labels:
        raise RuntimeError(
            "No se han construido secuencias. Revisa la ruta de datos, el modo de etiquetas o el tamaño de ventana."
        )

    x_hrv = np.asarray(hrv_sequences, dtype=np.float32)
    x_morph = np.asarray(morph_sequences, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    groups_arr = np.asarray(groups)

    return {
        "X_hrv": x_hrv,
        "X_morph": x_morph,
        "y": y,
        "groups": groups_arr,
        "meta": pd.DataFrame(meta_rows),
        "config": asdict(config),
        "class_names": dict(cfg.class_names),
        "feature_names_hrv": HRV_FEATURE_NAMES,
        "feature_names_morph": MORPH_FEATURE_NAMES,
    }


def concatenate_feature_branches(x_hrv: np.ndarray, x_morph: np.ndarray) -> np.ndarray:
    if x_hrv.shape[:2] != x_morph.shape[:2]:
        raise ValueError("x_hrv and x_morph must share sample and timestep dimensions.")
    return np.concatenate([x_hrv, x_morph], axis=-1).astype(np.float32)
