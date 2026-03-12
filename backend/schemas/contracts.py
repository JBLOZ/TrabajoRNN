from __future__ import annotations

HEALTH_RESPONSE = {
    "status": "ok",
    "service": "mitbih-recurrent-api",
    "available_models": ["v1", "v2", "v3"],
}

PREDICT_REQUEST_SCHEMA = {
    "type": "object",
    "required": ["model_version"],
    "properties": {
        "model_version": {"type": "string", "enum": ["v1", "v2", "v3"]},
        "input_mode": {"type": "string", "enum": ["precomputed_sequence", "raw_signal"], "default": "precomputed_sequence"},
        "sampling_rate_hz": {"type": "number"},
        "lead": {"type": "integer", "default": 0},
        "signal": {"type": "array", "items": {"type": "number"}},
        "r_peaks": {"type": "array", "items": {"type": "integer"}},
        "sequence_hrv": {"type": "array"},
        "sequence_morph": {"type": "array"},
        "sequence_fused": {"type": "array"},
        "return_xai": {"type": "boolean", "default": False},
    },
}

PREDICT_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "model_version": {"type": "string"},
        "predicted_class_index": {"type": "integer"},
        "predicted_class_name": {"type": "string"},
        "probabilities": {"type": "object"},
        "task": {"type": "string"},
        "xai": {"type": "object"},
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
}
