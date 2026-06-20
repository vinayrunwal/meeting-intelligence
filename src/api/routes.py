"""
Meeting Intelligence System — API Routes
========================================
REST endpoints for uploading audio, checking status, and streaming results.
"""


from flask import Blueprint, Response, current_app, jsonify, request, stream_with_context
from werkzeug.utils import secure_filename

from config.settings import settings
from src.api.error_handlers import APIError, ResourceNotFoundError, ValidationError
from src.api.streaming import stream_job_events
from src.api.validators import validate_upload
from src.utils.job_store import job_store

api_bp = Blueprint("api", __name__, url_prefix="/api/v1")


def _get_orchestrator():
    """Create the ML orchestrator only when a processing job needs it."""
    if current_app.orchestrator is None:
        try:
            from src.pipeline.orchestrator import PipelineOrchestrator
        except ModuleNotFoundError as exc:
            raise APIError(
                "Audio processing dependencies are not installed. "
                "Install the ML requirements before uploading audio.",
                status_code=503,
            ) from exc

        current_app.orchestrator = PipelineOrchestrator()
    return current_app.orchestrator


@api_bp.route("/upload", methods=["POST"])
def upload_audio():
    """
    Upload an audio file to start processing.
    Returns a job_id immediately while processing happens in the background.
    """
    if "file" not in request.files:
        raise ValidationError("No file part in the request")
        
    file = request.files["file"]
    validate_upload(file)
    orchestrator = _get_orchestrator()
    
    # Save file temporarily
    filename = secure_filename(file.filename)
    job_id = job_store.create_job(filename)
    
    # Ensure upload dir exists
    job_dir = settings.UPLOAD_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    file_path = job_dir / filename
    
    file.save(str(file_path))
    
    # Submit to background executor
    # current_app.executor is a ThreadPoolExecutor attached in app.py
    current_app.executor.submit(
        orchestrator.process_job,
        job_id=job_id,
        file_path=file_path
    )
    
    return jsonify({
        "success": True,
        "data": {
            "job_id": job_id,
            "message": "Audio uploaded successfully. Processing started.",
            "stream_url": f"/api/v1/stream/{job_id}",
            "result_url": f"/api/v1/result/{job_id}"
        }
    }), 202


@api_bp.route("/jobs", methods=["GET"])
def list_jobs():
    """List recent processing jobs."""
    limit = request.args.get("limit", 50, type=int)
    jobs = job_store.list_jobs(limit=limit)
    
    return jsonify({
        "success": True,
        "data": {
            "jobs": [j.model_dump() for j in jobs]
        }
    })


@api_bp.route("/result/<job_id>", methods=["GET"])
def get_result(job_id: str):
    """Get the final result of a completed job."""
    # First check if the job exists at all
    status = job_store.get_status(job_id)
    if not status:
        raise ResourceNotFoundError(f"Job {job_id} not found")
        
    result = job_store.get_result(job_id)
    if not result:
        # Job exists but result is not ready
        return jsonify({
            "success": True,
            "data": {
                "ready": False,
                "status": status.model_dump()
            }
        })
        
    return jsonify({
        "success": True,
        "data": {
            "ready": True,
            "result": result.model_dump()
        }
    })


@api_bp.route("/stream/<job_id>", methods=["GET"])
def stream_progress(job_id: str):
    """
    SSE endpoint to stream real-time progress for a job.
    Must be connected from an EventSource client.
    """
    status = job_store.get_status(job_id)
    if not status:
        raise ResourceNotFoundError(f"Job {job_id} not found")
        
    # Disable buffering for SSE to work properly through proxies
    headers = {
        'Cache-Control': 'no-cache',
        'X-Accel-Buffering': 'no',
        'Connection': 'keep-alive',
    }
    
    return Response(
        stream_with_context(stream_job_events(job_id)),
        mimetype="text/event-stream",
        headers=headers
    )


@api_bp.route("/jobs/<job_id>", methods=["DELETE"])
def cancel_job(job_id: str):
    """Cancel a running job or delete a completed job."""
    cancelled = job_store.cancel_job(job_id)
    if not cancelled:
        # Try deleting if already finished
        deleted = job_store.delete_job(job_id)
        if not deleted:
            raise ResourceNotFoundError(f"Job {job_id} not found or could not be removed")
            
        return jsonify({"success": True, "message": f"Job {job_id} deleted."})
        
    return jsonify({"success": True, "message": f"Job {job_id} cancelled."})
