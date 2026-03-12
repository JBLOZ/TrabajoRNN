from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import wfdb

from src.data.aami import AAMIConfig, PACED_RECORD_IDS


@dataclass
class BeatRecord:
    record_id: str
    signal: np.ndarray
    samples: np.ndarray
    symbols: np.ndarray
    fs: float


def list_record_ids(data_dir: str | Path) -> list[str]:
    data_dir = Path(data_dir)
    return sorted(p.stem for p in data_dir.glob("*.dat"))


def load_record(data_dir: str | Path, record_id: str, lead: int = 0) -> BeatRecord:
    data_dir = Path(data_dir)
    record = wfdb.rdrecord(str(data_dir / record_id))
    ann = wfdb.rdann(str(data_dir / record_id), "atr")
    signal = record.p_signal[:, lead].astype(np.float32)
    return BeatRecord(
        record_id=record_id,
        signal=signal,
        samples=np.asarray(ann.sample, dtype=np.int64),
        symbols=np.asarray(ann.symbol),
        fs=float(record.fs),
    )


def filter_valid_beats(
    beat_record: BeatRecord,
    label_mode: str = "aami_5",
    exclude_paced_records: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    if exclude_paced_records and beat_record.record_id in PACED_RECORD_IDS:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype="<U1")
    cfg = AAMIConfig(label_mode=label_mode)
    mask = np.array([s in cfg.valid_symbols for s in beat_record.symbols], dtype=bool)
    return beat_record.samples[mask], beat_record.symbols[mask]


def extract_centered_beat(
    signal: np.ndarray,
    center: int,
    window_before: int = 90,
    window_after: int = 162,
) -> np.ndarray | None:
    start = int(center) - int(window_before)
    end = int(center) + int(window_after)
    if start < 0 or end > len(signal):
        return None
    beat = signal[start:end]
    return beat.astype(np.float32, copy=False)


def safe_div(num: float, den: float, eps: float = 1e-8) -> float:
    return float(num) / float(den + eps)
