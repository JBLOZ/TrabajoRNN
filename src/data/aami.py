from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping

AAMI_5_CLASS_MAP: Dict[str, int] = {
    "N": 0, "L": 0, "R": 0, "e": 0, "j": 0,
    "A": 1, "a": 1, "J": 1, "S": 1,
    "V": 2, "E": 2,
    "F": 3,
    "/": 4, "f": 4, "Q": 4,
}

AAMI_4_CLASS_MAP: Dict[str, int] = {
    "N": 0, "L": 0, "R": 0, "e": 0, "j": 0,
    "A": 1, "a": 1, "J": 1, "S": 1,
    "V": 2, "E": 2,
    "F": 3,
}

CLASS_NAMES_5 = {
    0: "Normal (N)",
    1: "SVEB",
    2: "VEB",
    3: "Fusión (F)",
    4: "Desconocido (Q)",
}

CLASS_NAMES_4 = {
    0: "Normal (N)",
    1: "SVEB",
    2: "VEB",
    3: "Fusión (F)",
}

PACED_RECORD_IDS = {"102", "104", "107", "217"}
PACE_RELATED_SYMBOLS = {"/", "f"}


@dataclass(frozen=True)
class AAMIConfig:
    label_mode: str = "aami_5"

    @property
    def mapping(self) -> Mapping[str, int]:
        if self.label_mode == "aami_5":
            return AAMI_5_CLASS_MAP
        if self.label_mode == "aami_4":
            return AAMI_4_CLASS_MAP
        raise ValueError(f"Unsupported label_mode={self.label_mode}")

    @property
    def class_names(self) -> Mapping[int, str]:
        if self.label_mode == "aami_5":
            return CLASS_NAMES_5
        if self.label_mode == "aami_4":
            return CLASS_NAMES_4
        raise ValueError(f"Unsupported label_mode={self.label_mode}")

    @property
    def valid_symbols(self) -> set[str]:
        return set(self.mapping.keys())


def map_symbols(symbols: Iterable[str], label_mode: str = "aami_5") -> list[int]:
    cfg = AAMIConfig(label_mode=label_mode)
    return [cfg.mapping[s] for s in symbols if s in cfg.valid_symbols]
