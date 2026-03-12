from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from src.utils.io import load_joblib, load_json


@dataclass
class LoadedModelArtifacts:
    model: tf.keras.Model
    metadata: dict[str, Any]
    scaler_hrv: Any | None = None
    scaler_morph: Any | None = None
    scaler_fused: Any | None = None


def load_model_artifacts(model_dir: str | Path) -> LoadedModelArtifacts:
    model_dir = Path(model_dir)
    metadata = load_json(model_dir / "metadata.json")
    model = tf.keras.models.load_model(model_dir / "model.keras", compile=False)
    scaler_hrv = load_joblib(model_dir / "scaler_hrv.joblib") if (model_dir / "scaler_hrv.joblib").exists() else None
    scaler_morph = load_joblib(model_dir / "scaler_morph.joblib") if (model_dir / "scaler_morph.joblib").exists() else None
    scaler_fused = load_joblib(model_dir / "scaler_fused.joblib") if (model_dir / "scaler_fused.joblib").exists() else None
    return LoadedModelArtifacts(
        model=model,
        metadata=metadata,
        scaler_hrv=scaler_hrv,
        scaler_morph=scaler_morph,
        scaler_fused=scaler_fused,
    )


def predict_from_precomputed_sequences(
    artifacts: LoadedModelArtifacts,
    payload: dict[str, Any],
) -> dict[str, Any]:
    model = artifacts.model
    metadata = artifacts.metadata
    class_names = {int(k): v for k, v in metadata["class_names"].items()}

    if "sequence_hrv" in payload and "sequence_morph" in payload:
        x_hrv = np.asarray(payload["sequence_hrv"], dtype=np.float32)[None, ...]
        x_morph = np.asarray(payload["sequence_morph"], dtype=np.float32)[None, ...]
        if artifacts.scaler_hrv is not None:
            x_hrv = artifacts.scaler_hrv.transform(x_hrv)
        if artifacts.scaler_morph is not None:
            x_morph = artifacts.scaler_morph.transform(x_morph)
        inputs = {"hrv_input": x_hrv, "morph_input": x_morph}
    elif "sequence_fused" in payload:
        x = np.asarray(payload["sequence_fused"], dtype=np.float32)[None, ...]
        if artifacts.scaler_fused is not None:
            x = artifacts.scaler_fused.transform(x)
        inputs = x
    elif "sequence_hrv" in payload:
        x = np.asarray(payload["sequence_hrv"], dtype=np.float32)[None, ...]
        if artifacts.scaler_hrv is not None:
            x = artifacts.scaler_hrv.transform(x)
        inputs = x
    else:
        raise ValueError(
            "Payload sin secuencia válida. Usa sequence_hrv, sequence_fused o la dupla sequence_hrv + sequence_morph."
        )

    probs = model.predict(inputs, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = class_names[pred_idx]
    return {
        "predicted_class_index": pred_idx,
        "predicted_class_name": pred_label,
        "probabilities": {class_names[i]: float(probs[i]) for i in range(len(probs))},
        "model_version": metadata["model_version"],
        "task": metadata["task"],
    }
