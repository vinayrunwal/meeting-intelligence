"""
Meeting Intelligence System — Job Store
=========================================
Thread-safe in-memory job tracking with per-job SSE event queues.
Suitable for single-server deployments.
"""

from __future__ import annotations

import logging
import threading
import uuid
from datetime import datetime
from typing import Any, Optional

from src.models.schemas import (
    AudioMetadata,
    JobProgress,
    JobResult,
    JobState,
    JobStatus,
)

logger = logging.getLogger(__name__)


class JobStore:
    """
    Thread-safe in-memory store for processing jobs.

    Responsibilities:
        - Create and track job lifecycle
        - Store intermediate results
        - Provide SSE event broadcasting per job

    For production multi-server deployments, replace with Redis-backed store.
    """

    def __init__(self, max_jobs: int = 100) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, JobStatus] = {}
        self._results: dict[str, JobResult] = {}
        self._events: dict[str, list[dict[str, Any]]] = {}
        self._max_jobs = max_jobs

    def create_job(self, filename: str = "unknown") -> str:
        """
        Create a new processing job and return its ID.

        Returns
        -------
        str
            Unique job identifier (UUID4).
        """
        job_id = uuid.uuid4().hex[:12]
        now = datetime.utcnow()

        status = JobStatus(
            job_id=job_id,
            state=JobState.QUEUED,
            created_at=now,
            updated_at=now,
            progress=JobProgress(
                stage="queued",
                percent=0.0,
                message=f"Job created for: {filename}",
                timestamp=now,
            ),
        )

        with self._lock:
            # Evict oldest completed jobs if we exceed max
            if len(self._jobs) >= self._max_jobs:
                self._evict_completed()
            self._jobs[job_id] = status
            self._events[job_id] = []

        logger.info("Job created: %s for file: %s", job_id, filename)
        return job_id

    def update_progress(
        self,
        job_id: str,
        state: JobState,
        stage: str,
        percent: float,
        message: str = "",
    ) -> None:
        """
        Update a job's progress and broadcast an SSE event.

        Parameters
        ----------
        job_id : str
            The job identifier.
        state : JobState
            Current job state.
        stage : str
            Human-readable stage name.
        percent : float
            Progress percentage (0-100).
        message : str
            Optional status message.
        """
        now = datetime.utcnow()
        progress = JobProgress(
            stage=stage, percent=percent, message=message, timestamp=now,
        )

        with self._lock:
            if job_id not in self._jobs:
                logger.warning("Attempted to update unknown job: %s", job_id)
                return
            self._jobs[job_id].state = state
            self._jobs[job_id].progress = progress
            self._jobs[job_id].updated_at = now

            # Append SSE event
            event = {
                "type": "progress",
                "data": {
                    "job_id": job_id,
                    "state": state.value,
                    "stage": stage,
                    "percent": round(percent, 1),
                    "message": message,
                    "timestamp": now.isoformat(),
                },
            }
            self._events[job_id].append(event)

        logger.debug(
            "Job %s: %s — %.0f%% — %s", job_id, stage, percent, message,
        )

    def set_audio_metadata(self, job_id: str, metadata: AudioMetadata) -> None:
        """Attach audio metadata to a job."""
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].audio_metadata = metadata

    def store_result(self, job_id: str, result: JobResult) -> None:
        """Store the final processing result for a completed job."""
        with self._lock:
            self._results[job_id] = result
            if job_id in self._jobs:
                self._jobs[job_id].state = JobState.COMPLETED
                self._jobs[job_id].updated_at = datetime.utcnow()
                self._jobs[job_id].progress = JobProgress(
                    stage="completed",
                    percent=100.0,
                    message="Processing complete",
                    timestamp=datetime.utcnow(),
                )

                # Append completion event
                self._events[job_id].append({
                    "type": "complete",
                    "data": {
                        "job_id": job_id,
                        "state": "completed",
                        "processing_time": result.processing_time_seconds,
                    },
                })

        logger.info(
            "Job %s completed in %.1f seconds",
            job_id, result.processing_time_seconds,
        )

    def set_failed(self, job_id: str, error: str) -> None:
        """Mark a job as failed with an error message."""
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].state = JobState.FAILED
                self._jobs[job_id].error = error
                self._jobs[job_id].updated_at = datetime.utcnow()
                self._jobs[job_id].progress = JobProgress(
                    stage="failed",
                    percent=0,
                    message=error,
                    timestamp=datetime.utcnow(),
                )

                self._events[job_id].append({
                    "type": "error",
                    "data": {"job_id": job_id, "error": error},
                })

        logger.error("Job %s failed: %s", job_id, error)

    def get_status(self, job_id: str) -> Optional[JobStatus]:
        """Get the current status of a job."""
        with self._lock:
            return self._jobs.get(job_id)

    def get_result(self, job_id: str) -> Optional[JobResult]:
        """Get the final result of a completed job."""
        with self._lock:
            return self._results.get(job_id)

    def get_events(self, job_id: str, since_index: int = 0) -> list[dict[str, Any]]:
        """
        Get SSE events for a job since a given index.

        Parameters
        ----------
        job_id : str
            The job identifier.
        since_index : int
            Return events starting from this index.

        Returns
        -------
        list[dict]
            List of event dictionaries.
        """
        with self._lock:
            events = self._events.get(job_id, [])
            return events[since_index:]

    def list_jobs(
        self,
        state: Optional[JobState] = None,
        limit: int = 50,
    ) -> list[JobStatus]:
        """
        List jobs, optionally filtered by state.

        Parameters
        ----------
        state : JobState, optional
            Filter by state. None returns all.
        limit : int
            Maximum number of jobs to return.

        Returns
        -------
        list[JobStatus]
        """
        with self._lock:
            jobs = list(self._jobs.values())

        if state is not None:
            jobs = [j for j in jobs if j.state == state]

        # Sort by creation time, newest first
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return jobs[:limit]

    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a queued or processing job.

        Returns True if the job was cancelled, False if it was already
        completed/failed or does not exist.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return False
            if job.state in (JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED):
                return False
            job.state = JobState.CANCELLED
            job.updated_at = datetime.utcnow()
            return True

    def delete_job(self, job_id: str) -> bool:
        """Remove a job and all its data from the store."""
        with self._lock:
            removed = job_id in self._jobs
            self._jobs.pop(job_id, None)
            self._results.pop(job_id, None)
            self._events.pop(job_id, None)
            return removed

    def _evict_completed(self) -> None:
        """Remove oldest completed jobs to make room. Must hold lock."""
        completed = [
            (jid, j) for jid, j in self._jobs.items()
            if j.state in (JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED)
        ]
        completed.sort(key=lambda x: x[1].created_at)

        evicted = 0
        for jid, _ in completed:
            if len(self._jobs) - evicted < self._max_jobs:
                break
            del self._jobs[jid]
            self._results.pop(jid, None)
            self._events.pop(jid, None)
            evicted += 1

        if evicted:
            logger.info("Evicted %d completed jobs from store", evicted)


# Module-level singleton
job_store = JobStore()
