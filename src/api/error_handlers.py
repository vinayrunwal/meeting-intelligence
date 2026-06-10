"""
Meeting Intelligence System — API Error Handlers
================================================
Centralised error handling for the Flask API.
Converts exceptions into structured JSON responses.
"""

from __future__ import annotations

import logging
from typing import Any, Tuple

from flask import Blueprint, current_app, jsonify
from werkzeug.exceptions import HTTPException

logger = logging.getLogger(__name__)

error_bp = Blueprint("errors", __name__)


class APIError(Exception):
    """Base exception for API errors."""
    def __init__(self, message: str, status_code: int = 400, payload: Any = None):
        super().__init__()
        self.message = message
        self.status_code = status_code
        self.payload = payload

    def to_dict(self) -> dict[str, Any]:
        rv = dict(self.payload or ())
        rv["error"] = self.message
        rv["success"] = False
        return rv


class ValidationError(APIError):
    """Raised when request payload or file validation fails."""
    def __init__(self, message: str, payload: Any = None):
        super().__init__(message, status_code=400, payload=payload)


class ResourceNotFoundError(APIError):
    """Raised when a job or resource is not found."""
    def __init__(self, message: str = "Resource not found"):
        super().__init__(message, status_code=404)


@error_bp.app_errorhandler(APIError)
def handle_api_error(error: APIError) -> Tuple[Any, int]:
    """Handle custom API exceptions."""
    logger.warning("API Error [%d]: %s", error.status_code, error.message)
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response, error.status_code


@error_bp.app_errorhandler(HTTPException)
def handle_http_exception(error: HTTPException) -> Tuple[Any, int]:
    """Handle standard Werkzeug/Flask HTTP exceptions."""
    response = jsonify({
        "success": False,
        "error": error.description,
        "code": error.code
    })
    response.status_code = error.code or 500
    return response, response.status_code


@error_bp.app_errorhandler(Exception)
def handle_generic_exception(error: Exception) -> Tuple[Any, int]:
    """Handle unexpected server errors."""
    logger.exception("Unhandled Server Error: %s", str(error))
    
    # Don't leak stack traces in production
    msg = "An unexpected server error occurred."
    if current_app.debug:
        msg = str(error)
        
    response = jsonify({
        "success": False,
        "error": msg,
        "type": error.__class__.__name__
    })
    response.status_code = 500
    return response, 500
