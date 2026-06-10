"""
Meeting Intelligence System — Flask App Factory
===============================================
Initialises the Flask application, registers blueprints, 
and sets up the background task executor.
"""

import atexit
import logging
from concurrent.futures import ThreadPoolExecutor

from flask import Flask, jsonify
from flask_cors import CORS

from config.logging_config import configure_logging
from config.settings import settings
from src.api.error_handlers import error_bp
from src.api.routes import api_bp
from src.pipeline.orchestrator import PipelineOrchestrator

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
    CORS(app, resources={r"/api/*": {"origins": settings.api.cors_origins}})
    
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
    
    # Initialize the ML orchestrator
    # We initialize it here so it lives with the app lifecycle
    app.orchestrator = PipelineOrchestrator()
    
    # Cleanup executor on exit
    def cleanup():
        logger.info("Shutting down executor...")
        app.executor.shutdown(wait=False)
        
    atexit.register(cleanup)
    
    # 6. Add Health Check
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
