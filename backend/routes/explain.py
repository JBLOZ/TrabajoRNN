from __future__ import annotations

from flask import Blueprint, current_app, jsonify, request

from backend.services.explain_service import explain

explain_bp = Blueprint("explain", __name__)


@explain_bp.post("/explain")
def explain_route():
    payload = request.get_json(force=True, silent=False)
    registry = current_app.config["MODEL_REGISTRY"]
    response = explain(payload, registry)
    return jsonify(response)
