"""
Meeting Intelligence System — SSE Streaming
===========================================
Generator functions for Server-Sent Events (SSE) to stream job progress.
"""

import json
import logging
import time
from typing import Generator

from src.models.schemas import JobState
from src.utils.job_store import job_store

logger = logging.getLogger(__name__)


def stream_job_events(job_id: str) -> Generator[str, None, None]:
    """
    Generate an SSE stream for a specific job.
    Yields events as they occur until the job completes or fails.
    
    Format:
    data: {"stage": "transcription", "percent": 15.0} \n\n
    """
    logger.info("Client connected to stream for job %s", job_id)
    
    # First, verify the job exists
    job = job_store.get_status(job_id)
    if not job:
        yield f"data: {json.dumps({'error': 'Job not found'})}\n\n"
        return

    # Send initial state immediately
    yield f"data: {json.dumps({'stage': job.progress.stage, 'percent': job.progress.percent, 'message': job.progress.message})}\n\n"

    last_event_index = 0
    poll_interval = 0.5  # seconds
    timeout_counter = 0
    max_timeout = 7200  # 2 hours max connection

    try:
        while True:
            # Check for new events in the store
            events = job_store.get_events(job_id, since_index=last_event_index)
            
            for event in events:
                # Yield in SSE format
                yield f"data: {json.dumps(event['data'])}\n\n"
                last_event_index += 1
                
                # If complete or error, terminate the stream
                if event["type"] in ("complete", "error"):
                    logger.info("Stream for job %s terminating naturally", job_id)
                    return

            # Check if job was removed or stuck
            job = job_store.get_status(job_id)
            if not job:
                yield f"data: {json.dumps({'error': 'Job was removed'})}\n\n"
                return
                
            if job.state in (JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED):
                # We missed the terminal event somehow
                return

            # Avoid tight loop
            time.sleep(poll_interval)
            
            # Send a ping every 15 seconds to keep connection alive
            timeout_counter += poll_interval
            if timeout_counter % 15 < poll_interval:
                yield ": ping\n\n"
                
            if timeout_counter > max_timeout:
                logger.warning("Stream for job %s timed out", job_id)
                yield f"data: {json.dumps({'error': 'Stream timeout'})}\n\n"
                return

    except GeneratorExit:
        logger.info("Client disconnected from stream for job %s", job_id)
    except Exception as e:
        logger.error("Error in SSE stream for job %s: %s", job_id, e)
        yield f"data: {json.dumps({'error': 'Internal stream error'})}\n\n"
