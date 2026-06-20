"""
Lightweight dev server for Meeting Intelligence.
Starts a minimal Flask app that exposes `/health` without loading ML components.
Use this when you want to run the API quickly without heavy ML dependencies.
"""
import os
from flask import Flask, jsonify, send_file

from config.settings import settings, PROJECT_ROOT
from src.api.cors import configure_cors

app = Flask(__name__)
# Allow cross-origin requests for the frontend running on a different port
configure_cors(app, settings.api.cors_origins)

@app.route("/")
def index():
    """Serve the frontend interface."""
    frontend_dir = PROJECT_ROOT / "frontend"
    return send_file(frontend_dir / "index.html")

@app.route("/health")
def health():
    return jsonify({"status": "lightweight", "gpu_available": False})

if __name__ == "__main__":
    app.run(host=settings.api.host, port=settings.api.port, debug=True)
