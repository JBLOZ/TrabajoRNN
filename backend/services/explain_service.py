from __future__ import annotations

from typing import Any

import numpy as np

from backend.services.artifact_loader import ModelRegistry
from backend.services.preprocess_service import validate_predict_payload
from src.inference.pipeline import predict_from_precomputed_sequences
from src.xai.gradients import integrated_gradients, saliency_map
from src.xai.occlusion import temporal_occlusion_importance


def explain(payload: dict[str, Any], registry: ModelRegistry) -> dict[str, Any]:
    validate_predict_payload(payload)
    artifacts = registry.load(payload["model_version"])

    if "sequence_hrv" in payload and "sequence_morph" in payload:
        x_hrv = np.asarray(payload["sequence_hrv"], dtype=np.float32)[None, ...]
        x_morph = np.asarray(payload["sequence_morph"], dtype=np.float32)[None, ...]
        if artifacts.scaler_hrv is not None:
            x_hrv = artifacts.scaler_hrv.transform(x_hrv)
        if artifacts.scaler_morph is not None:
            x_morph = artifacts.scaler_morph.transform(x_morph)
        model_inputs = {"hrv_input": x_hrv, "morph_input": x_morph}
    elif "sequence_fused" in payload:
        x = np.asarray(payload["sequence_fused"], dtype=np.float32)[None, ...]
        if artifacts.scaler_fused is not None:
            x = artifacts.scaler_fused.transform(x)
        model_inputs = x
    elif "sequence_hrv" in payload:
        x = np.asarray(payload["sequence_hrv"], dtype=np.float32)[None, ...]
        if artifacts.scaler_hrv is not None:
            x = artifacts.scaler_hrv.transform(x)
        model_inputs = x
    else:
        raise ValueError("Payload no compatible con explain().")

    prediction = predict_from_precomputed_sequences(artifacts, payload)
    class_index = prediction["predicted_class_index"]

    sal = saliency_map(artifacts.model, model_inputs, class_index=class_index)
    ig = integrated_gradients(artifacts.model, model_inputs, class_index=class_index)
    occ = temporal_occlusion_importance(artifacts.model, model_inputs, class_index=class_index)

    return {
        "prediction": prediction,
        "xai": {
            "saliency": _to_serializable(sal),
            "integrated_gradients": _to_serializable(ig),
            "occlusion": _to_serializable(occ),
        },
    }


def _to_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
        return obj.item()
    return obj
