"""
Meeting Intelligence System — Flask App Factory
===============================================
Initialises the Flask application, registers blueprints, 
and sets up the background task executor.
"""

import atexit
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from flask import Flask, jsonify

from config.logging_config import configure_logging
from config.settings import settings, PROJECT_ROOT
from src.api.cors import configure_cors
from src.api.error_handlers import error_bp
from src.api.routes import api_bp

logger = logging.getLogger("api")


def create_app() -> Flask:
    """Flask application factory."""
    # 1. Setup logging
    configure_logging()
    
    # 2. Create app
    app = Flask(__name__)
    
    # Configure limits
    app.config['MAX_CONTENT_LENGTH'] = settings.audio.max_upload_size_mb * 1024 * 1024
    
    # 3. Setup CORS
    configure_cors(app, settings.api.cors_origins, resources={
        r"/api/*": {"origins": settings.api.cors_origins},
        r"/health": {"origins": settings.api.cors_origins},
    })
    
    # 4. Register blueprints
    app.register_blueprint(error_bp)
    app.register_blueprint(api_bp)
    
    # 5. Initialize background executor
    # We use ThreadPoolExecutor because the heavy ML inference releases the GIL
    # or runs in separate processes internally (e.g. CTranslate2).
    # Single worker is recommended for GPU to avoid VRAM exhaustion.
    workers = settings.api.max_concurrent_jobs
    logger.info("Initializing ThreadPoolExecutor with %d workers", workers)
    app.executor = ThreadPoolExecutor(max_workers=workers)
    
    # The ML orchestrator is loaded lazily on first upload so health/status
    # endpoints can run in lightweight environments without ML dependencies.
    app.orchestrator = None
    
    # Cleanup executor on exit
    def cleanup():
        try:
            app.executor.shutdown(wait=False)
        except Exception:
            pass
        
    atexit.register(cleanup)
    
    # 6. Add Frontend and Health Check
    from flask import send_file
    @app.route("/")
    def index():
        frontend_dir = PROJECT_ROOT / "frontend"
        return send_file(frontend_dir / "index.html")
        
    @app.route("/health")
    def health_check():
        return jsonify({
            "status": "healthy",
            "gpu_available": False, # Will update dynamically in prod
        })
        
    logger.info("Flask application initialized")
    return app


if __name__ == "__main__":
    app = create_app()
    app.run(
        host=settings.api.host, 
        port=settings.api.port, 
        debug=settings.api.debug
    )
