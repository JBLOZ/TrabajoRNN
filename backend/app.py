from __future__ import annotations

import os
from pathlib import Path

from flask import Flask
from flask_cors import CORS

from backend.routes.explain import explain_bp
from backend.routes.health import health_bp
from backend.routes.models import models_bp
from backend.routes.predict import predict_bp
from backend.services.artifact_loader import ModelRegistry


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)

    registry_path = os.environ.get(
        "MODEL_REGISTRY_PATH",
        str(Path(__file__).resolve().parent / "model_registry" / "registry.json"),
    )
    app.config["MODEL_REGISTRY"] = ModelRegistry(registry_path)

    app.register_blueprint(health_bp)
    app.register_blueprint(models_bp)
    app.register_blueprint(predict_bp)
    app.register_blueprint(explain_bp)
    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
