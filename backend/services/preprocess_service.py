from __future__ import annotations

from typing import Any


def validate_predict_payload(payload: dict[str, Any]) -> None:
    if "model_version" not in payload:
        raise ValueError("Falta model_version.")
    if payload.get("input_mode", "precomputed_sequence") == "precomputed_sequence":
        if not any(k in payload for k in ("sequence_hrv", "sequence_morph", "sequence_fused")):
            raise ValueError(
                "Para input_mode='precomputed_sequence' debes enviar sequence_hrv, sequence_fused o sequence_hrv+sequence_morph."
            )
    else:
        if "signal" not in payload:
            raise ValueError("Para input_mode='raw_signal' debes enviar signal.")
        if "r_peaks" not in payload:
            raise ValueError(
                "El backend esqueleto no implementa todavía detección QRS; envía r_peaks o precomputed_sequence."
            )
