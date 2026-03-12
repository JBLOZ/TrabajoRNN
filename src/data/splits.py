from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


DE_CHAZAL_DS1 = [
    "101", "106", "108", "109", "112", "114", "115", "116", "118", "119", "122",
    "124", "201", "203", "205", "207", "208", "209", "215", "220", "223", "230",
]
DE_CHAZAL_DS2 = [
    "100", "103", "105", "111", "113", "117", "121", "123", "200", "202", "210",
    "212", "213", "214", "219", "221", "222", "228", "231", "232", "233", "234",
]


@dataclass
class SplitIndices:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return {"train_idx": self.train_idx, "val_idx": self.val_idx, "test_idx": self.test_idx}


def _has_required_classes(y: np.ndarray, indices: np.ndarray, required_classes: Iterable[int]) -> bool:
    present = set(np.unique(y[indices]).tolist())
    return set(required_classes).issubset(present)


def _score_split(y: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray) -> float:
    overall = np.bincount(y) / len(y)
    score = 0.0
    for idx in [train_idx, val_idx, test_idx]:
        dist = np.bincount(y[idx], minlength=len(overall)) / max(len(idx), 1)
        score += float(np.abs(dist - overall).sum())
    return score


def make_fixed_grouped_splits(
    y: np.ndarray,
    groups: np.ndarray,
    seed: int = 42,
    required_classes: Iterable[int] | None = None,
    outer_splits: int = 5,
    inner_splits: int = 4,
    max_seed_tries: int = 50,
) -> SplitIndices:
    """
    Create one train/val/test split using StratifiedGroupKFold while ensuring
    group disjointness and class coverage.
    """
    y = np.asarray(y)
    groups = np.asarray(groups)
    required_classes = list(np.unique(y) if required_classes is None else required_classes)

    best: SplitIndices | None = None
    best_score = float("inf")

    for trial_seed in range(seed, seed + max_seed_tries):
        outer = StratifiedGroupKFold(n_splits=outer_splits, shuffle=True, random_state=trial_seed)
        for trainval_idx, test_idx in outer.split(np.zeros(len(y)), y, groups):
            if not _has_required_classes(y, trainval_idx, required_classes):
                continue
            if not _has_required_classes(y, test_idx, required_classes):
                continue

            y_trainval = y[trainval_idx]
            groups_trainval = groups[trainval_idx]
            inner = StratifiedGroupKFold(n_splits=inner_splits, shuffle=True, random_state=trial_seed + 777)

            for train_sub_idx, val_sub_idx in inner.split(np.zeros(len(y_trainval)), y_trainval, groups_trainval):
                train_idx = trainval_idx[train_sub_idx]
                val_idx = trainval_idx[val_sub_idx]

                if not _has_required_classes(y, train_idx, required_classes):
                    continue
                if not _has_required_classes(y, val_idx, required_classes):
                    continue

                if set(groups[train_idx]) & set(groups[val_idx]):
                    continue
                if set(groups[train_idx]) & set(groups[test_idx]):
                    continue
                if set(groups[val_idx]) & set(groups[test_idx]):
                    continue

                candidate = SplitIndices(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
                candidate_score = _score_split(y, train_idx, val_idx, test_idx)
                if candidate_score < best_score:
                    best = candidate
                    best_score = candidate_score

        if best is not None:
            return best

    raise RuntimeError(
        "No se pudo construir un split grouped+stratified con cobertura de clases. "
        "Reduce clases, usa otro seed o ajusta el protocolo."
    )


def predefined_de_chazal_split(groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    groups = np.asarray(groups).astype(str)
    train_mask = np.isin(groups, DE_CHAZAL_DS1)
    test_mask = np.isin(groups, DE_CHAZAL_DS2)
    return np.where(train_mask)[0], np.where(test_mask)[0]
