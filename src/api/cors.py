"""CORS setup with a lightweight fallback for local environments."""

from __future__ import annotations

from typing import Any

try:
    from flask_cors import CORS as FlaskCORS
except ModuleNotFoundError:
    FlaskCORS = None


def configure_cors(app: Any, origins: str = "*", resources: dict[str, Any] | None = None) -> None:
    """Configure CORS without making flask-cors mandatory for lightweight runs.

    Note: `flask-cors` expects `resources` to be a specific mapping shape.
    Some parts of this project pass a dict; if that shape is unexpected,
    we gracefully fall back to a simple after_request hook.
    """
    if FlaskCORS is not None:
        try:
            FlaskCORS(app, resources=resources)
            return
        except ValueError:
            # Fallback for unexpected `resources` argument shape
            pass


    @app.after_request
    def add_cors_headers(response):
        response.headers.setdefault("Access-Control-Allow-Origin", origins)
        response.headers.setdefault("Access-Control-Allow-Headers", "Content-Type, Authorization")
        response.headers.setdefault("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        return response
