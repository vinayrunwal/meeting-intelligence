#!/usr/bin/env python3
"""
Meeting Intelligence System — Latency Profiler
==============================================
Profiles end-to-end execution time for the ML pipeline stages.
"""

import logging
import time
import tracemalloc
from pathlib import Path
from contextlib import contextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@contextmanager
def profile_block(name: str, metrics_dict: dict):
    """Context manager to profile execution time and memory of a block."""
    start_time = time.time()
    tracemalloc.start()
    
    yield
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    duration = time.time() - start_time
    
    metrics_dict[name] = {
        "duration_seconds": duration,
        "peak_memory_mb": peak / 10**6
    }
    
    logger.info(f"[{name}] Completed in {duration:.2f}s | Peak RAM: {peak / 10**6:.2f}MB")


def profile_pipeline(audio_path: str | Path):
    """
    Run a full profile of the orchestrator pipeline.
    This is a mock implementation that would normally instantiate the orchestrator.
    """
    from src.pipeline.orchestrator import PipelineOrchestrator
    
    logger.info(f"Starting latency profile for: {audio_path}")
    metrics = {}
    
    # Normally we would patch the orchestrator to record times,
    # or just run it and time the whole thing.
    
    with profile_block("End-to-End Pipeline", metrics):
        orchestrator = PipelineOrchestrator()
        try:
            # We use a dummy job_id
            orchestrator.process_job("profile_job_001", Path(audio_path))
        except Exception as e:
            logger.error(f"Pipeline failed during profiling: {e}")
            
    return metrics
