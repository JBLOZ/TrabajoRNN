from __future__ import annotations

from flask import Blueprint, current_app, jsonify, request

from backend.services.predict_service import predict

predict_bp = Blueprint("predict", __name__)


@predict_bp.post("/predict")
def predict_route():
    payload = request.get_json(force=True, silent=False)
    registry = current_app.config["MODEL_REGISTRY"]
    response = predict(payload, registry)
    return jsonify(response)
