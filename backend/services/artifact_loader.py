from __future__ import annotations

import json
from pathlib import Path

from src.inference.pipeline import load_model_artifacts


class ModelRegistry:
    def __init__(self, registry_path: str | Path):
        self.registry_path = Path(registry_path)
        self.registry = json.loads(self.registry_path.read_text(encoding="utf-8"))

    def list_models(self) -> dict:
        return self.registry

    def get_model_dir(self, model_version: str) -> Path:
        if model_version not in self.registry:
            raise KeyError(f"Modelo no registrado: {model_version}")
        relative = self.registry[model_version]["artifacts"]["model_dir"]
        return (self.registry_path.parent.parent.parent / relative).resolve()

    def load(self, model_version: str):
        return load_model_artifacts(self.get_model_dir(model_version))
