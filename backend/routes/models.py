from __future__ import annotations

from flask import Blueprint, current_app, jsonify

models_bp = Blueprint("models", __name__)


@models_bp.get("/models")
def list_models():
    registry = current_app.config["MODEL_REGISTRY"]
    return jsonify(registry.list_models())
