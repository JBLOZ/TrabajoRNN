from __future__ import annotations

from typing import Any

from backend.services.artifact_loader import ModelRegistry
from backend.services.preprocess_service import validate_predict_payload
from src.inference.pipeline import predict_from_precomputed_sequences


def predict(payload: dict[str, Any], registry: ModelRegistry) -> dict[str, Any]:
    validate_predict_payload(payload)
    model_version = payload["model_version"]
    artifacts = registry.load(model_version)
    response = predict_from_precomputed_sequences(artifacts, payload)
    response["warnings"] = []
    if payload.get("input_mode", "precomputed_sequence") == "raw_signal":
        response["warnings"].append(
            "El esqueleto actual solo soporta inferencia efectiva con secuencias ya preprocesadas o con r_peaks externos."
        )
    return response
